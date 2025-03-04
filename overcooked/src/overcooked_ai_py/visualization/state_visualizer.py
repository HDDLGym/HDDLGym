import copy
import math
import os
import random
import pygame

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.layout_generator import (
    COUNTER,
    DISH_DISPENSER,
    EMPTY,
    ONION_DISPENSER,
    POT,
    SERVING_LOC,
    TOMATO_DISPENSER,
)
from overcooked_ai_py.static import FONTS_DIR, GRAPHICS_DIR
from overcooked_ai_py.utils import (
    classproperty,
    cumulative_rewards_from_rew_list,
    generate_temporary_file_path,
)
from overcooked_ai_py.visualization.pygame_utils import (
    MultiFramePygameImage,
    blit_on_new_surface_of_size,
    run_static_resizeable_window,
    scale_surface_by_factor,
    vstack_surfaces,
)
from overcooked_ai_py.visualization.visualization_utils import (
    show_image_in_ipython,
    show_ipython_images_slider,
)

roboto_path = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")



class StateVisualizer:
    TERRAINS_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "terrain.png"),
        os.path.join(GRAPHICS_DIR, "terrain.json"),
    )
    OBJECTS_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "objects.png"),
        os.path.join(GRAPHICS_DIR, "objects.json"),
    )
    SOUPS_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "soups.png"),
        os.path.join(GRAPHICS_DIR, "soups.json"),
    )
    SOUPS_KEY_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "soups_key.png"),
        os.path.join(GRAPHICS_DIR, "soups_key.json"),
    )
    CHEFS_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "chefs.png"),
        os.path.join(GRAPHICS_DIR, "chefs.json"),
    )
    ARROW_IMG = pygame.image.load(os.path.join(GRAPHICS_DIR, "arrow.png"))
    INTERACT_IMG = pygame.image.load(
        os.path.join(GRAPHICS_DIR, "interact.png")
    )

    COIN_IMG = pygame.transform.scale(
        pygame.image.load(os.path.join(GRAPHICS_DIR, "coin.png")),
        (230, 230))

    CLOCK_IMG = pygame.transform.scale(
        pygame.image.load(os.path.join(GRAPHICS_DIR, "clock.png")),
        (230, 230))

    STAY_IMG = pygame.image.load(os.path.join(GRAPHICS_DIR, "stay.png"))
    ICON_SERVING_IMG = pygame.transform.scale(
        pygame.image.load(os.path.join(GRAPHICS_DIR, "icon_serving.png")),
        (200, 200))
    UNSCALED_TILE_SIZE = 256
    DEFAULT_VALUES = {
        "height": None,  # if None use grid_width - NOTE: can chop down hud if hud is wider than grid
        "width": None,  # if None use (hud_height+grid_height)
        "tile_size": 256,
        "window_fps": 30,
        "player_colors": ["blue", "green"],
        "is_rendering_hud": True,
        "hud_font_size": 100,
        "hud_font_path": roboto_path,
        "hud_system_font_name": None,  # if set to None use hud_font_path
        "hud_font_color": (0, 0, 0),  # white
        "hud_data_default_key_order": [
            "all_orders",
            "bonus_orders",
            "time_left",
            "score",
            "potential",
        ],
        "hud_interline_size": 10,
        "hud_margin_bottom": 0,
        "hud_margin_top": 0,
        "hud_margin_left": 10,
        "hud_distance_between_orders": 10,
        "hud_order_size": 150,
        "is_rendering_cooking_timer": True,
        "show_timer_when_cooked": True,
        "cooking_timer_font_size": 60,  # # if set to None use cooking_timer_font_path
        "cooking_timer_font_path": roboto_path,
        "cooking_timer_system_font_name": None,
        "cooking_timer_font_color": (255, 0, 0),  # red
        "grid": None,
        "background_color": (100,100,100), # (155, 101, 0),  # color of empty counter
        "is_rendering_action_probs": True,  # whatever represent visually on the grid what actions some given agent would make
    }
    TILE_TO_FRAME_NAME = {
        EMPTY: "floor",
        COUNTER: "counter",
        ONION_DISPENSER: "onions",
        TOMATO_DISPENSER: "tomatoes",
        POT: "pot",
        DISH_DISPENSER: "dishes",
        SERVING_LOC: "serve",
    }

    def __init__(self, **kwargs):
        params = copy.deepcopy(self.DEFAULT_VALUES)
        params.update(kwargs)
        self.configure(**params)
        self.reload_fonts()

    def reload_fonts(self):
        pygame.font.init()
        if not hasattr(self, "_font"):
            self._fonts = {}
        # initializing fonts only if needed because it can take a quite long time,
        #   see https://pygame.readthedocs.io/en/latest/4_text/text.html#initialize-a-font
        if self.is_rendering_hud:
            self.hud_font = self._init_font(
                self.hud_font_size,
                self.hud_font_path,
                self.hud_system_font_name,
            )
        else:
            self.hud_font = None

        if self.is_rendering_cooking_timer:
            self.cooking_timer_font = self._init_font(
                self.cooking_timer_font_size,
                self.cooking_timer_font_path,
                self.cooking_timer_system_font_name,
            )
        else:
            self.cooking_timer_font = None

    @classmethod
    def configure_defaults(cls, **kwargs):
        cls._check_config_validity(kwargs)
        cls.DEFAULT_VALUES.update(copy.deepcopy(kwargs))

    def configure(self, **kwargs):
        StateVisualizer._check_config_validity(kwargs)
        for param_name, param_value in copy.deepcopy(kwargs).items():
            setattr(self, param_name, param_value)

    @staticmethod
    def default_hud_data(state, **kwargs):
        result = {
            "timestep": state.timestep,
            "all_orders": [r.to_dict() for r in state.all_orders],
            "bonus_orders": [r.to_dict() for r in state.bonus_orders],
        }
        result.update(copy.deepcopy(kwargs))
        return result

    @staticmethod
    def default_hud_data_from_trajectories(trajectories, trajectory_idx=0):
        scores = cumulative_rewards_from_rew_list(
            trajectories["ep_rewards"][trajectory_idx]
        )
        return [
            StateVisualizer.default_hud_data(state, score=scores[i])
            for i, state in enumerate(
                trajectories["ep_states"][trajectory_idx]
            )
        ]

    def display_rendered_trajectory(
        self,
        trajectories,
        trajectory_idx=0,
        hud_data=None,
        action_probs=None,
        img_directory_path=None,
        img_extension=".png",
        img_prefix="",
        ipython_display=True,
    ):
        """
        saves images of every timestep from trajectory in img_directory_path (or temporary directory if not path is not specified)
        trajectories (dict): trajectories dict, same format as used by AgentEvaluator
        trajectory_idx(int): index of trajectory in case of multiple trajectories inside trajectories param
        img_path (str): img_directory_path - path to directory where consequtive images will be saved
        ipython_display(bool): if True render slider with rendered states
        hud_data(list(dict)): hud data for every timestep
        action_probs(list(list((list(float))))): action probs for every player and timestep acessed in the way action_probs[timestep][player][action]
        """
        states = trajectories["ep_states"][trajectory_idx]
        grid = trajectories["mdp_params"][trajectory_idx]["terrain"]
        if hud_data is None:
            if self.is_rendering_hud:
                hud_data = StateVisualizer.default_hud_data_from_trajectories(
                    trajectories, trajectory_idx
                )
            else:
                hud_data = [None] * len(states)

        if action_probs is None:
            action_probs = [None] * len(states)

        if not img_directory_path:
            img_directory_path = generate_temporary_file_path(
                prefix="overcooked_visualized_trajectory", extension=""
            )
        os.makedirs(img_directory_path, exist_ok=True)
        img_pathes = []
        for i, state in enumerate(states):
            img_name = img_prefix + str(i) + img_extension
            img_path = os.path.join(img_directory_path, img_name)
            img_pathes.append(
                self.display_rendered_state(
                    state=state,
                    hud_data=hud_data[i],
                    action_probs=action_probs[i],
                    grid=grid,
                    img_path=img_path,
                    ipython_display=True,
                    window_display=False,
                )
            )

        if ipython_display:
            return show_ipython_images_slider(img_pathes, "timestep")

        return img_directory_path

    def display_rendered_state(
        self,
        state,
        hud_data=None,
        action_probs=None,
        grid=None,
        img_path=None,
        ipython_display=False,
        window_display=False,
    ):
        """
        renders state as image
        state (OvercookedState): state to render
        hud_data (dict): dict with hud data, keys are used for string that describes after using _key_to_hud_text on them
        grid (iterable): 2d map of the layout, when not supplied take grid from object attribute NOTE: when grid in both method param and object atribute is no supplied it will raise an error
        img_path (str): if it is not None save image to specific path
        ipython_display (bool): if True render state in ipython cell, if img_path is None create file with randomized name in /tmp directory
        window_display (bool): if True render state into pygame window
        action_probs(list(list(float))): action probs for every player acessed in the way action_probs[player][action]
        """
        assert (
            window_display or img_path or ipython_display
        ), "specify at least one of the ways to output result state image: window_display, img_path, or ipython_display"
        surface = self.render_state(
            state, grid, hud_data, action_probs=action_probs
        )

        if img_path is None and ipython_display:
            img_path = generate_temporary_file_path(
                prefix="overcooked_visualized_state_", extension=".png"
            )

        if img_path is not None:
            pygame.image.save(surface, img_path)

        if ipython_display:
            show_image_in_ipython(img_path)

        if window_display:
            run_static_resizeable_window(surface, self.window_fps)

        return img_path

    def render_state(self, state, grid, hud_data=None, action_probs=None, agent_traces=None):
        """
        returns surface with rendered game state scaled to selected size,
        decoupled from display_rendered_state function to make testing easier
        """
        pygame.init()
        grid = grid or self.grid
        assert grid
        grid_surface = pygame.surface.Surface(
            self._unscaled_grid_pixel_size(grid)
        )
        self._render_grid(grid_surface, grid)
        self.grid = grid_surface
        self._render_players(grid_surface, state.players)
        self._render_objects(grid_surface, state.objects, grid)

        # self.mark_tiles(grid_surface, grid)

        if self.scale_by_factor != 1:
            grid_surface = scale_surface_by_factor(
                grid_surface, self.scale_by_factor
            )

        if agent_traces is not None:
            agent_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
            self._render_trace(grid_surface, agent_traces, agent_colors)


        # self._render_grid2(grid_surface, grid)

        # render text after rescaling as text looks bad when is rendered small resolution and then rescalled to bigger one
        if self.is_rendering_cooking_timer:
            self._render_cooking_timers(grid_surface, state.objects, grid)

        # arrows does not seem good when rendered in very small resolution
        if self.is_rendering_action_probs and action_probs is not None:
            self._render_actions_probs(
                grid_surface, state.players, action_probs
            )

        if self.is_rendering_hud and hud_data:
            hud_width = self.width or grid_surface.get_width()

            hud_height = self.grid.get_height() + 200
            hud_surface = pygame.surface.Surface(
                (hud_width, hud_height), pygame.SRCALPHA
            )
            hud_surface.fill((0, 0, 0, 0))  # Transparent background for HUD

            # self._render_hud_data(hud_surface, hud_data)

            grid_surface.blit(hud_surface, (0, 0))  # Blit HUD on top of the grid_surface
            rendered_surface = grid_surface

        else:
            hud_width = None
            rendered_surface = grid_surface

        result_surface_size = (
            self.width or rendered_surface.get_width(),
            self.height or rendered_surface.get_height(),
        )

        #############################################################
        # HUD Strip Setup
        hud_strip_height = 170
        hud_strip_surface = pygame.surface.Surface((self.width or grid_surface.get_width(), hud_strip_height))
        hud_strip_surface.fill((230,180,83))

        # Combine surfaces
        total_height = (self.height or grid_surface.get_height()) + hud_strip_height
        combined_surface = pygame.surface.Surface((self.width or grid_surface.get_width(), total_height))

        # Blit game (and HUD) below the strip
        combined_surface.blit(hud_strip_surface, (0, 0))  # Blit the strip at the very top

        # 3. Blit the game (including any HUD) below the strip
        combined_surface.blit(grid_surface, (0, hud_strip_height))


        self._render_hud_data(hud_surface, hud_data)
        combined_surface.blit(hud_surface, (0, 0))        

        #############################################################


        if result_surface_size != rendered_surface.get_size():
            result_surface = blit_on_new_surface_of_size(
                rendered_surface,
                result_surface_size,
                background_color=self.background_color,
            )
        else:
            result_surface = rendered_surface

        return combined_surface
        # return result_surface

    def _render_trace(self, surface, agent_traces, agent_colors):
        """
        Render the trace of where agents have been as a line connecting the positions.
        agent_traces: Dictionary where keys are agent indices and values are lists of positions.
        agent_colors: Dictionary where keys are agent indices and values are colors.
        """
        for agent_idx, trace in agent_traces.items():
            color = agent_colors[agent_idx]
            if len(trace) > 1:
                for i in range(1, len(trace)):
                    start_pos = self._position_in_scaled_pixels(trace[i-1])
                    end_pos = self._position_in_scaled_pixels(trace[i])
                    start_pos_centered = (start_pos[0] + self.tile_size // 2, start_pos[1] + self.tile_size // 2)
                    end_pos_centered = (end_pos[0] + self.tile_size // 2, end_pos[1] + self.tile_size // 2)
                    pygame.draw.line(surface, color, start_pos_centered, end_pos_centered, width=5)

    def _render_trace(self, surface, agent_traces, agent_colors):
        """
        Render the trace of where agents have been as a line connecting the positions.
        agent_traces: Dictionary where keys are agent indices and values are lists of positions.
        agent_colors: Dictionary where keys are agent indices and values are colors.
        """
        for agent_idx, trace in agent_traces.items():
            color = agent_colors[agent_idx]
            if len(trace) > 1:
                for i in range(1, len(trace)):
                    start_pos = self._position_in_scaled_pixels(trace[i-1])
                    end_pos = self._position_in_scaled_pixels(trace[i])
                    start_pos_centered = (start_pos[0] + self.tile_size // 2, start_pos[1] + self.tile_size // 2)
                    end_pos_centered = (end_pos[0] + self.tile_size // 2, end_pos[1] + self.tile_size // 2)
                    pygame.draw.line(surface, color, start_pos_centered, end_pos_centered, width=5)

    @property
    def scale_by_factor(self):
        return self.tile_size / StateVisualizer.UNSCALED_TILE_SIZE

    @property
    def hud_line_height(self):
        return self.hud_interline_size + self.hud_font_size

    @staticmethod
    def _check_config_validity(config):
        assert set(config.keys()).issubset(
            set(StateVisualizer.DEFAULT_VALUES.keys())
        )

    def _init_font(self, font_size, font_path=None, system_font_name=None):
        if system_font_name:
            key = "%i-sys:%s" % (font_size, system_font_name)
            font = self._fonts.get(key) or pygame.font.SysFont(
                system_font_name, font_size
            )
        else:
            key = "%i-path:%s" % (font_size, font_path)
            font = self._fonts.get(key) or pygame.font.Font(
                font_path, font_size
            )
        self._fonts[key] = font
        return font

    def _unscaled_grid_pixel_size(self, grid):
        y_tiles = len(grid)
        x_tiles = len(grid[0])
        return (
            x_tiles * self.UNSCALED_TILE_SIZE,
            y_tiles * self.UNSCALED_TILE_SIZE,
        )

    # def _render_grid(self, surface, grid):
    #     for y_tile, row in enumerate(grid):
    #         for x_tile, tile in enumerate(row):
    #             self.TERRAINS_IMG.blit_on_surface(
    #                 surface,
    #                 self._position_in_unscaled_pixels((x_tile, y_tile)),
    #                 StateVisualizer.TILE_TO_FRAME_NAME[tile],
    #             )

    def _draw_divider_around_tile(self, surface, tile_position):
        """
        Draws a visual divider (like a wall or a line) around the given tile using a rectangle.
        Fills the inside of the rectangle with a light brown color.
        """
        # Get the position in pixels of the tile
        (x_pixel, y_pixel) = self._position_in_scaled_pixels(tile_position)

        # Define the colors and thickness
        fill_color = (168,133,65) # (205, 133, 63)  # Light brown color (RGB)
        divider_color = (108,78,54) # (0, 0, 0)    # Black color for the divider
        divider_thickness = 5        # Thickness of the divider

        # Draw a filled rectangle (inside of the tile)
        rect = pygame.Rect(x_pixel, y_pixel, self.tile_size, self.tile_size)
        pygame.draw.rect(surface, fill_color, rect)

        # Draw the border around the tile (outer edge)
        pygame.draw.rect(surface, divider_color, rect, divider_thickness)

    def _is_adjacent_to_floor(self, grid, x_tile, y_tile):
        """
        Check if the given tile is adjacent to a floor tile (EMPTY).
        """
        # Define the possible directions to check (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dy, dx)

        for dy, dx in directions:
            adj_y, adj_x = y_tile + dy, x_tile + dx
            # Ensure we're within the grid bounds
            if 0 <= adj_y < len(grid) and 0 <= adj_x < len(grid[0]):
                # If any adjacent tile is a floor (EMPTY), return True
                if StateVisualizer.TILE_TO_FRAME_NAME[grid[adj_y][adj_x]] == 'floor':
                    return True
        return False

    def _render_grid(self, surface, grid):
        for y_tile, row in enumerate(grid):
            for x_tile, tile in enumerate(row):
                self.TERRAINS_IMG.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels((x_tile, y_tile)),
                    StateVisualizer.TILE_TO_FRAME_NAME[tile],
                )

                if StateVisualizer.TILE_TO_FRAME_NAME[tile] == 'serve':
                    self._render_on_tile_position(surface, self.ICON_SERVING_IMG, (x_tile, y_tile), horizontal_align="center", vertical_align="center")
                
                if StateVisualizer.TILE_TO_FRAME_NAME[tile] == 'counter' and self._is_adjacent_to_floor(grid, x_tile, y_tile):
                    self._draw_divider_around_tile(surface, (x_tile, y_tile))

                # Check if tile exists in the image set
                frame_name = StateVisualizer.TILE_TO_FRAME_NAME.get(tile, None)
                if frame_name is None:
                    print(f"Warning: Tile frame '{tile}' not found!")
                    continue
        
    def _render_grid2(self, surface, grid):
        for y_tile, row in enumerate(grid):
            for x_tile, tile in enumerate(row):

                # Calculate position in scaled pixels
                # position = self._position_in_unscaled_pixels((x_tile, y_tile))
                position = self._position_in_scaled_pixels((x_tile, y_tile))
                # position = (position[0] - 2, position[1])

                # Draw grid outline with smaller line thickness
                pygame.draw.rect(
                    surface,
                    (0, 0, 0),  # Outline color (black)
                    pygame.Rect(
                        position,
                        (self.tile_size, self.tile_size)
                    ),
                    4
                )

    def _position_in_unscaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x, y) = position
        return (self.UNSCALED_TILE_SIZE * x, self.UNSCALED_TILE_SIZE * y)

    def _position_in_scaled_pixels(self, position):
        """
        get x and y coordinates in tiles, returns x and y coordinates in pixels
        """
        (x, y) = position
        return (self.tile_size * x, self.tile_size * y)

    def _render_players(self, surface, players):
        def chef_frame_name(direction_name, held_object_name):
            frame_name = direction_name
            if held_object_name:
                frame_name += "-" + held_object_name
            return frame_name

        def chef_frame_name2(direction_name, held_object_name, player_color_name):
            if player_color_name == "blue":
                part1 = "chef1"
            else: 
                part1 = "chef2"
            part2 = direction_name.lower()
            part3 = held_object_name
            new_part3 = part3.replace("-", "_")
            frame_name = f"{part1}_{part2}"
            if held_object_name:
                frame_name += "_" + new_part3
            return frame_name


        def hat_frame_name(direction_name, player_color_name):
            return "%s-%shat" % (direction_name, player_color_name)


        for player_num, player in enumerate(players):
            player_color_name = self.player_colors[player_num]
            direction_name = Direction.DIRECTION_TO_NAME[player.orientation]

            held_obj = player.held_object
            if held_obj is None:
                held_object_name = ""
            else:
                if held_obj.name == "soup":
                    if "onion" in held_obj.ingredients:
                        held_object_name = "soup-onion"
                    else:
                        held_object_name = "soup-tomato"
                else:
                    held_object_name = held_obj.name

            ###############################################3
            # a = StateVisualizer
            # surface = scale_surface_by_factor(surface, a.scale_by_factor)
            def scale_rect(rect, scale_factor):
                return pygame.Rect(
                    rect.x * scale_factor,
                    rect.y * scale_factor,
                    rect.width * scale_factor,
                    rect.height * scale_factor
                )
            
            # scale_factor = 10  # Change this to your desired scale factor

            # for key in self.CHEFS_IMG.frames_rectangles:
            #     self.CHEFS_IMG.frames_rectangles[key] = scale_rect(
            #         self.CHEFS_IMG.frames_rectangles[key],
            #         scale_factor
            #     )
            ###############################################3

            # self.CHEFS_IMG.image = pygame.transform.scale(self.CHEFS_IMG.image, (400,400))

            self.CHEFS_IMG.blit_on_surface(
                surface,
                self._position_in_unscaled_pixels(player.position),
                chef_frame_name2(direction_name, held_object_name, player_color_name),
            )

            # self.CHEFS_IMG.blit_on_surface(
            #     surface,
            #     self._position_in_unscaled_pixels(player.position),
            #     chef_frame_name(direction_name, held_object_name),
            # )
            # self.CHEFS_IMG.blit_on_surface(
            #     surface,
            #     self._position_in_unscaled_pixels(player.position),
            #     hat_frame_name(direction_name, player_color_name),
            # )

    @staticmethod
    def _soup_frame_name(ingredients_names, status):
        num_onions = ingredients_names.count("onion")
        num_tomatoes = ingredients_names.count("tomato")
        return "soup_%s_tomato_%i_onion_%i" % (
            status,
            num_tomatoes,
            num_onions,
        )

    def _render_objects(self, surface, objects, grid):
        def render_soup(surface, obj, grid):
            (x_pos, y_pos) = obj.position
            if grid[y_pos][x_pos] == POT:
                if obj.is_ready:
                    soup_status = "cooked"
                else:
                    soup_status = "idle"
            else:  # grid[x][y] != POT
                soup_status = "done"
            frame_name = StateVisualizer._soup_frame_name(
                obj.ingredients, soup_status
            )
            self.SOUPS_IMG.blit_on_surface(
                surface,
                self._position_in_unscaled_pixels(obj.position),
                frame_name,
            )

        for obj in objects.values():
            if obj.name == "soup":
                render_soup(surface, obj, grid)
            else:
                self.OBJECTS_IMG.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels(obj.position),
                    obj.name,
                )

    def _render_cooking_timers(self, surface, objects, grid):
        for key, obj in objects.items():
            (x_pos, y_pos) = obj.position
            if obj.name == "soup" and grid[y_pos][x_pos] == POT:
                if obj._cooking_tick != -1 and (
                    obj._cooking_tick <= obj.cook_time
                    or self.show_timer_when_cooked
                ):
                    render_time = (obj.cook_time - obj._cooking_tick) / 10
                    if render_time == 0: render_time = 0

                    text_surface = self.cooking_timer_font.render(
                        str(render_time),
                        True,
                        self.cooking_timer_font_color,
                    )
                    # text_surface = self.cooking_timer_font.render(
                    #     str(obj._cooking_tick),
                    #     True,
                    #     self.cooking_timer_font_color,
                    # )
                    (tile_pos_x, tile_pos_y) = self._position_in_scaled_pixels(
                        obj.position
                    )

                    # calculate font position to be in center on x axis, and 0.9 from top on y axis
                    font_position = (
                        tile_pos_x
                        + int(
                            (self.tile_size - text_surface.get_width()) * 0.5
                        ),
                        tile_pos_y
                        + int(
                            (self.tile_size - text_surface.get_height()) * 0.9
                        ),
                    )
                    surface.blit(text_surface, font_position)

    def _sorted_hud_items(self, hud_data):
        def default_order_then_alphabetic(item):
            key = item[0]
            try:
                i = self.hud_data_default_key_order.index(key)
            except:
                i = 99999
            return (i, key)

        return sorted(hud_data.items(), key=default_order_then_alphabetic)

    def _key_to_hud_text(self, key):
        return key.replace("_", " ").title() + ": "


    def _calculate_hud_height(self, hud_data):
        return (
            self.hud_margin_top
            + len(hud_data) * self.hud_line_height
            + self.hud_margin_bottom
        )

    def _render_on_tile_position(
        self,
        scaled_grid_surface,
        source_surface,
        tile_position,
        horizontal_align="left",
        vertical_align="top",
    ):
        assert vertical_align in ["top", "center", "bottom"]
        left_x, top_y = self._position_in_scaled_pixels(tile_position)
        if horizontal_align == "left":
            x = left_x
        elif horizontal_align == "center":
            x = left_x + (self.tile_size - source_surface.get_width()) / 2
        elif horizontal_align == "right":
            x = left_x + self.tile_size - source_surface.get_width()
        else:
            raise ValueError(
                "horizontal_align can have one of the values: "
                + str(["left", "center", "right"])
            )

        if vertical_align == "top":
            y = top_y
        elif vertical_align == "center":
            y = top_y + (self.tile_size - source_surface.get_height()) / 2
        elif vertical_align == "bottom":
            y = top_y + self.tile_size - source_surface.get_height()
        else:
            raise ValueError(
                "vertical_align can have one of the values: "
                + str(["top", "center", "bottom"])
            )

        scaled_grid_surface.blit(source_surface, (x, y))

    def _render_actions_probs(self, surface, players, action_probs):
        direction_to_rotation = {
            Direction.NORTH: 0,
            Direction.WEST: 90,
            Direction.SOUTH: 180,
            Direction.EAST: 270,
        }
        direction_to_aligns = {
            Direction.NORTH: {
                "horizontal_align": "center",
                "vertical_align": "bottom",
            },
            Direction.WEST: {
                "horizontal_align": "right",
                "vertical_align": "center",
            },
            Direction.SOUTH: {
                "horizontal_align": "center",
                "vertical_align": "top",
            },
            Direction.EAST: {
                "horizontal_align": "left",
                "vertical_align": "center",
            },
        }

        rescaled_arrow = pygame.transform.scale(
            self.ARROW_IMG, (self.tile_size, self.tile_size)
        )
        # divide width by math.sqrt(2) to always fit both interact icon and stay icon into single tile
        rescaled_interact = pygame.transform.scale(
            self.INTERACT_IMG,
            (int(self.tile_size / math.sqrt(2)), self.tile_size),
        )
        rescaled_stay = pygame.transform.scale(
            self.STAY_IMG, (int(self.tile_size / math.sqrt(2)), self.tile_size)
        )
        for player, probs in zip(players, action_probs):
            if probs is not None:
                for action in Action.ALL_ACTIONS:
                    # use math sqrt to make probability proportional to area of the image
                    size = math.sqrt(probs[Action.ACTION_TO_INDEX[action]])
                    if action == "interact":
                        img = pygame.transform.rotozoom(
                            rescaled_interact, 0, size
                        )
                        self._render_on_tile_position(
                            surface,
                            img,
                            player.position,
                            horizontal_align="left",
                            vertical_align="center",
                        )
                    elif action == Action.STAY:
                        img = pygame.transform.rotozoom(rescaled_stay, 0, size)
                        self._render_on_tile_position(
                            surface,
                            img,
                            player.position,
                            horizontal_align="right",
                            vertical_align="center",
                        )
                    else:
                        position = Action.move_in_direction(
                            player.position, action
                        )
                        img = pygame.transform.rotozoom(
                            rescaled_arrow, direction_to_rotation[action], size
                        )
                        self._render_on_tile_position(
                            surface,
                            img,
                            position,
                            **direction_to_aligns[action]
                        )

#################################################################################

    def _render_hud_data(self, surface, hud_data):
        def hud_text_position(line_num):
            return (
                self.hud_margin_left,
                self.hud_margin_top + self.hud_line_height * line_num,
            )

        def hud_recipes_position(text_surface, text_surface_position):
            (text_surface_x, text_surface_y) = text_surface_position
            return (text_surface_x + text_surface.get_width(), text_surface_y)

        def get_hud_recipes_surface(orders_dicts, bonus_order):
            # order_width = order_height = self.hud_order_size
            order_height = self.hud_order_size 
            order_width = 220
            scaled_order_size = (order_width, order_width)
            scaled_order_size = (order_width, order_height)
            orders_surface_height = order_height
            orders_surface_width = (
                len(orders_dicts) * order_width
                + (len(orders_dicts) - 1) * self.hud_distance_between_orders
            )


            unscaled_order_size = (
                self.UNSCALED_TILE_SIZE * 1.5 * 1.4,
                self.UNSCALED_TILE_SIZE * 1.5 * 1.4,
            )

            recipes_surface = pygame.surface.Surface((orders_surface_width * 5, orders_surface_height), pygame.SRCALPHA)
            recipes_surface.fill((0, 0, 0, 0))
            next_surface_x = 0
            next_surface_rect = -10 + 20
            i = 0
            for order_dict in orders_dicts:
                i += 1
                #################################
                # Define box colors
                box_background_color = (255,255,255)# (168,133,65)
                box_border_color = (0,0,0)
                box_thickness = 5

                if i != 1:
                    next_surface_rect += 530

                box_rect = pygame.Rect(
                    next_surface_rect,
                    0,
                    256 * 1.5,
                    order_height - box_thickness*2
                )

                pygame.draw.rect(recipes_surface, box_background_color, box_rect)
                pygame.draw.rect(recipes_surface, box_border_color, box_rect, box_thickness)

                #################################

                frame_name = StateVisualizer._soup_frame_name(
                    order_dict["ingredients"], "done"
                )
                # unscaled_order_surface = pygame.surface.Surface(
                #     unscaled_order_size
                # )
                unscaled_order_surface = pygame.surface.Surface(
                    unscaled_order_size, pygame.SRCALPHA  # Ensure transparency support
                )                
                # unscaled_order_surface.fill(self.background_color)
                unscaled_order_surface.fill((0,0,0,0))
                self.SOUPS_KEY_IMG.blit_on_surface(
                    unscaled_order_surface, (60, -70), frame_name
                )
               
                if scaled_order_size == unscaled_order_size:
                    scaled_order_surface = unscaled_order_surface
                else:
                    scaled_order_surface = pygame.transform.scale(
                        unscaled_order_surface, (order_width, order_width)
                    )
                recipes_surface.blit(scaled_order_surface, (next_surface_x, 0))
                next_surface_x += (
                    order_width + self.hud_distance_between_orders
                ) + 300

                #####################################

                total_points = 0
                for item in order_dict["ingredients"]:
                    if item == 'onion':
                        total_points += 2
                    elif item == 'tomato':
                        total_points += 1

                if order_dict["ingredients"] == bonus_order:
                    total_points *= 2
                

                ################################################################
                points_text = f"{total_points}"
                soup_points_font_size = 80
                soup_points_font = self._init_font(soup_points_font_size, self.hud_font_path)
                text_surface = soup_points_font.render(points_text, True, self.hud_font_color)
                scaled_coin = pygame.transform.scale(self.COIN_IMG, (120, 120))
                text_position = (next_surface_rect + 260 + 65 - 50, 21)
                coin_position = (
                    text_position[0] + (text_surface.get_width() - scaled_coin.get_width()) // 2 + 0,  # Horizontally center the coin under the text
                    10  # Fixed offset (e.g., 110 pixels) below the text
                )
                recipes_surface.blit(scaled_coin, coin_position)
                recipes_surface.blit(text_surface, text_position)


                ################################################################

            return recipes_surface

        # Iterate over the sorted hud_data items
        for hud_line_num, (key, value) in enumerate(self._sorted_hud_items(hud_data)):
            hud_text = self._key_to_hud_text(key)
            if key not in ["all_orders", "bonus_orders", "start_all_orders", "start_bonus_orders"]:
                hud_text += str(value)

            if key in ["bonus_orders"]: continue

            # Render the text onto the surface
            text_surface_position = hud_text_position(hud_line_num)
            if key == 'net_reward':
                # text_surface_position = (90,1100)
                text_surface_position = (90,self.grid.get_height()-20)
                hud_text = str(int(hud_text[11:]))

                text_width = text_surface.get_width()
                text_height = text_surface.get_height()
                coin_position = (text_surface_position[0] - text_width / 2 + self.COIN_IMG.get_width() / 2 + 40, text_surface_position[1] - self.COIN_IMG.get_height() / 2 + text_height / 2)  # Adjust y-offset as needed
                surface.blit(self.COIN_IMG, coin_position)                
            if key == 'time left':
                text_surface_position = (self.grid.get_width()-182,self.grid.get_height()-20)
                hud_text = str(int(hud_text[10:]))

                text_width = text_surface.get_width()
                text_height = text_surface.get_height()
                clock_position = (text_surface_position[0] - text_width / 2 + self.CLOCK_IMG.get_width() / 2 - 145, text_surface_position[1] - self.CLOCK_IMG.get_height() / 2 + text_height / 2)  # Adjust y-offset as needed
                surface.blit(self.CLOCK_IMG, clock_position)                

            text_surface = self.hud_font.render(hud_text, True, self.hud_font_color)

            if not key == 'all_orders':
                surface.blit(text_surface, text_surface_position)

            # Render the order images for specific keys
            if key in ["all_orders", "start_all_orders", "start_bonus_orders"] and value:
                text_surface_position = (-450,10)
                recipes_surface_position = hud_recipes_position(text_surface, text_surface_position)

                first_bonus_order = None
                if hud_data['bonus_orders']:
                    first_bonus_order = hud_data['bonus_orders'][0]['ingredients']
                recipes_surface = get_hud_recipes_surface(value, first_bonus_order)
                # assert (
                #     recipes_surface.get_width() + text_surface.get_width()
                #     <= surface.get_width()
                # ), "surface width is too small to fit recipes in a single line"

                surface.blit(recipes_surface, recipes_surface_position)


    def _calculate_hud_height(self, hud_data):
        return (
            self.hud_margin_top
            + len(hud_data) * self.hud_line_height
            + self.hud_margin_bottom
        )
