import math

import numpy as np

from jaxmarl.viz.window import Window
import jaxmarl.viz.grid_rendering as rendering
from jaxmarl.environments.overcooked_v2.common import OBJECT_TO_INDEX, COLOR_TO_INDEX, COLORS
from jaxmarl.environments.overcooked_v2.overcooked import COOKING_TIMES


INDEX_TO_COLOR = [k for k,v in COLOR_TO_INDEX.items()]
TILE_PIXELS = 32

COLOR_TO_AGENT_INDEX = {0:0, 2:1} # Hardcoded. Red is first, blue is second

def generate_status_index():
    pot_statuses = {}
    current_index = 0
    pot_statuses['empty'] = current_index
    combinations = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (0, 2), (0, 3),
        (1, 1), (2, 1), (1, 2)
    ]
    
    for (onion_count, tomato_count) in combinations:
        ready_index = current_index + 1
        in_pot_index = ready_index + 1 + COOKING_TIMES['onion'] * onion_count + COOKING_TIMES['tomato'] * tomato_count
        
        pot_statuses[f'{onion_count}_onion_{tomato_count}_tomato_ready'] = ready_index
        pot_statuses[f'{onion_count}_onion_{tomato_count}_tomato_in_pot'] = in_pot_index

        current_index = in_pot_index

    return pot_statuses

STATUSES_INDEX = generate_status_index()

def generate_lookup_index():
    pot_statuses = {}
    current_index = 0
    pot_statuses[current_index] = (0, 0)  # empty
    combinations = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (0, 2), (0, 3),
        (1, 1), (2, 1), (1, 2)
    ]
    
    for (onion_count, tomato_count) in combinations:
        base_index = current_index
        ready_index = base_index + 1
        max_time_index = base_index + COOKING_TIMES['onion'] * onion_count + COOKING_TIMES['tomato'] * tomato_count
        in_pot_index = max_time_index + 2
        
        pot_statuses[ready_index] = (onion_count, tomato_count)  # ready
        pot_statuses[in_pot_index] = (onion_count, tomato_count)  # in pot
        
        current_index = in_pot_index

    return pot_statuses

def update_status_with_labels(status_dict, status_index):
    updated_status_dict = {}
    reverse_status_index = {v: k for k, v in status_index.items()}
    
    for key, value in status_dict.items():
        status_label = reverse_status_index.get(key, "unknown")
        status_type = '_'.join(status_label.split('_')[4:])
        updated_status_dict[key] = (value, status_type)
    
    updated_status_dict[0] = ((0, 0), 'empty')
    return updated_status_dict


def get_pot_components(pot_status, updated_status_dict):
    pot_status_below = nearest_below(pot_status)
    time_difference = pot_status - pot_status_below
    status_tuple = updated_status_dict[pot_status_below]
    onion_count, tomato_count, status_type = status_tuple[0][0], status_tuple[0][1], status_tuple[1] 

    is_cooking = status_type not in ["in_pot", "empty"]
    is_done = time_difference == 0 and status_type == "ready"
    max_time = COOKING_TIMES['onion'] * onion_count + COOKING_TIMES['tomato'] * tomato_count

    return onion_count, tomato_count, time_difference, is_cooking, is_done, max_time

status_dict = generate_lookup_index()
keys_list = list(status_dict.keys())
nearest_below = lambda given_number: max([x for x in keys_list if x <= given_number], default=None)
updated_status_dict = update_status_with_labels(status_dict, STATUSES_INDEX)


class OvercookedVisualizer:
    """
    Manages a window and renders contents of EnvState instances to it.
    """
    tile_cache = {}

    def __init__(self):
        self.window = None

    def _lazy_init_window(self):
        if self.window is None:
            self.window = Window('minimax')

    def show(self, block=False):
        self._lazy_init_window()
        self.window.show(block=block)

    def render(self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        return self._render_state(agent_view_size, state, highlight, tile_size)

    def animate(self, state_seq, agent_view_size, filename="animation.gif"):
        """Animate a gif give a state sequence and save if to file."""
        import imageio

        padding = agent_view_size - 2  # show

        def get_frame(state):
            grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
            # Render the state
            frame = OvercookedVisualizer._render_grid(
                grid,
                tile_size=TILE_PIXELS,
                highlight_mask=None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv
            )
            return frame

        frame_seq =[get_frame(state) for state in state_seq]

        imageio.mimsave(filename, frame_seq, 'GIF', duration=0.5)


    def render_grid(self, grid, tile_size=TILE_PIXELS, k_rot90=0, agent_dir_idx=None):
        self._lazy_init_window()

        img = OvercookedVisualizer._render_grid(
                grid,
                tile_size,
                highlight_mask=None,
                agent_dir_idx=agent_dir_idx,
            )
        # img = np.transpose(img, axes=(1,0,2))
        if k_rot90 > 0:
            img = np.rot90(img, k=k_rot90)

        self.window.show_img(img)

    def _render_state(self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the state
        """
        self._lazy_init_window()

        padding = agent_view_size-2 # show
        grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        grid_offset = np.array([1,1])
        h, w = grid.shape[:2]
        # === Compute highlight mask
        highlight_mask = np.zeros(shape=(h,w), dtype=bool)

        if highlight:
            # TODO: Fix this for multiple agents
            f_vec = state.agent_dir
            r_vec = np.array([-f_vec[1], f_vec[0]])

            fwd_bound1 = state.agent_pos
            fwd_bound2 = state.agent_pos + state.agent_dir*(agent_view_size-1)
            side_bound1 = state.agent_pos - r_vec*(agent_view_size//2)
            side_bound2 = state.agent_pos + r_vec*(agent_view_size//2)

            min_bound = np.min(np.stack([
                        fwd_bound1,
                        fwd_bound2,
                        side_bound1,
                        side_bound2]) + grid_offset, 0)

            min_y = min(max(min_bound[1], 0), highlight_mask.shape[0]-1)
            min_x = min(max(min_bound[0],0), highlight_mask.shape[1]-1)

            max_y = min(max(min_bound[1]+agent_view_size, 0), highlight_mask.shape[0]-1)
            max_x = min(max(min_bound[0]+agent_view_size, 0), highlight_mask.shape[1]-1)

            highlight_mask[min_y:max_y,min_x:max_x] = True

        # Render the whole grid
        img = OvercookedVisualizer._render_grid(
            grid,
            tile_size,
            highlight_mask=highlight_mask if highlight else None,
            agent_dir_idx=state.agent_dir_idx,
            agent_inv=state.agent_inv
        )
        self.window.show_img(img)

    @classmethod
    def _render_obj(
        cls,
        obj,
        img):
        # Render each kind of object
        obj_type = obj[0]
        color = INDEX_TO_COLOR[obj[1]]

        def render_dish(img, num_onions, num_tomatoes):
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS[color])

            # Coordinates for up to 3 items
            coords = [(0.35, 0.5), (0.5, 0.5), (0.65, 0.5)]
            
            # Render onions
            for i in range(num_onions):
                onion_fn = rendering.point_in_circle(coords[i][0], coords[i][1], 0.13)
                rendering.fill_coords(img, onion_fn, COLORS["orange"])

            # Render tomatoes
            for i in range(num_tomatoes):
                tomato_fn = rendering.point_in_circle(coords[i + num_onions][0], coords[i + num_onions][1], 0.13)
                rendering.fill_coords(img, tomato_fn, COLORS["red"])


        if obj_type == OBJECT_TO_INDEX['wall']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['goal']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            rendering.fill_coords(img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['agent']:
            agent_dir_idx = obj[2]
            tri_fn = rendering.point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )
            # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
            direction_reording = [3,1,0,2]
            direction = direction_reording[agent_dir_idx]
            tri_fn = rendering.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*direction)
            rendering.fill_coords(img, tri_fn, COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['empty']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['onion_pile']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            onion_fns = [rendering.point_in_circle(*coord, 0.15) for coord in [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35),
                                                                              (0.4, 0.8), (0.75, 0.75)]]
            [rendering.fill_coords(img, onion_fn, COLORS[color]) for onion_fn in onion_fns]
        elif obj_type == OBJECT_TO_INDEX['tomato_pile']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            tomato_fns = [rendering.point_in_circle(*coord, 0.15) for coord in [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35),
                                                                              (0.4, 0.8), (0.75, 0.75)]]
            [rendering.fill_coords(img, tomato_fn, COLORS[color]) for tomato_fn in tomato_fns]
        elif obj_type == OBJECT_TO_INDEX['onion']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            onion_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
            rendering.fill_coords(img, onion_fn, COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['tomato']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            tomato_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
            rendering.fill_coords(img, tomato_fn, COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['plate_pile']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            plate_fns = [rendering.point_in_circle(*coord, 0.2) for coord in [(0.3, 0.3), (0.75, 0.42),
                                                                              (0.4, 0.75)]]
            [rendering.fill_coords(img, plate_fn, COLORS[color]) for plate_fn in plate_fns]
        elif obj_type == OBJECT_TO_INDEX['plate']:
            rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS[color])
        elif obj_type == OBJECT_TO_INDEX['dish_0']:
            render_dish(img, num_onions=1, num_tomatoes=0)
        elif obj_type == OBJECT_TO_INDEX['dish_1']:
            render_dish(img, num_onions=2, num_tomatoes=0)
        elif obj_type == OBJECT_TO_INDEX['dish_2']:
            render_dish(img, num_onions=3, num_tomatoes=0)
        elif obj_type == OBJECT_TO_INDEX['dish_3']:
            render_dish(img, num_onions=0, num_tomatoes=1)
        elif obj_type == OBJECT_TO_INDEX['dish_4']:
            render_dish(img, num_onions=0, num_tomatoes=2)												
        elif obj_type == OBJECT_TO_INDEX['dish_5']:
            render_dish(img, num_onions=0, num_tomatoes=3)												
        elif obj_type == OBJECT_TO_INDEX['dish_6']:
            render_dish(img, num_onions=1, num_tomatoes=1)												
        elif obj_type == OBJECT_TO_INDEX['dish_7']:
            render_dish(img, num_onions=2, num_tomatoes=1)												
        elif obj_type == OBJECT_TO_INDEX['dish_8']:
            render_dish(img, num_onions=1, num_tomatoes=2)												
        elif obj_type == OBJECT_TO_INDEX['pot']:
            OvercookedVisualizer._render_pot(obj, img)
            # rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])
            # pot_fns = [rendering.point_in_rect(0.1, 0.9, 0.3, 0.9),
            # 		   rendering.point_in_rect(0.1, 0.9, 0.20, 0.23),
            # 		   rendering.point_in_rect(0.4, 0.6, 0.15, 0.20),]
            # [rendering.fill_coords(img, pot_fn, COLORS[color]) for pot_fn in pot_fns]
        else:
            raise ValueError(f'Rendering object at index {obj_type} is currently unsupported.')

    @classmethod
    def _render_pot(
        cls,
        obj,
        img):
        pot_status = obj[-1]


        num_onions, num_tomatoes, time_to_done, is_cooking, is_done, max_time = get_pot_components(pot_status, updated_status_dict)
        progress = (max_time - time_to_done) / max_time if max_time > 0 else 0

        # num_onions = np.max([23-pot_status, 0])
        # is_cooking = np.array((pot_status < 20) * (pot_status > 0))
        # is_done = np.array(pot_status == 0)

        ### !!! SORT THIS !!!

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

        # Render onions in pot
        if num_onions > 0 and not is_done:
            onion_fns = [rendering.point_in_circle(*coord, 0.13) for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]]
            onion_fns = onion_fns[:num_onions]
            [rendering.fill_coords(img, onion_fn, COLORS["yellow"]) for onion_fn in onion_fns]
            if not is_cooking:
                lid_fn = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
                handle_fn =rendering.rotate_fn(handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)

        if num_tomatoes > 0 and not is_done:
            tomato_fns = [rendering.point_in_circle(*coord, 0.13) for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]]
            tomato_fns = tomato_fns[num_onions:num_onions+num_tomatoes]
            [rendering.fill_coords(img, tomato_fn, COLORS["red"]) for tomato_fn in tomato_fns]
            if not is_cooking:
                lid_fn = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
                handle_fn =rendering.rotate_fn(handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)

        # Render done soup
        if is_done:
            soup_fn = rendering.point_in_rect(0.12, 0.88, 0.23, 0.35)
            rendering.fill_coords(img, soup_fn, COLORS["orange"])

        # Render the pot itself
        pot_fns = [pot_fn, lid_fn, handle_fn]
        [rendering.fill_coords(img, pot_fn, COLORS["black"]) for pot_fn in pot_fns]

        # Render progress bar
        if is_cooking:
            # progress_fn = rendering.point_in_rect(0.1, 0.9-(0.9-0.1)/20*pot_status, 0.83, 0.88)
            progress_fn = rendering.point_in_rect(0.1, 0.1 + 0.8 * progress, 0.83, 0.88)			
            rendering.fill_coords(img, progress_fn, COLORS["green"])


    @classmethod
    def _render_inv(
        cls,
        obj,
        img):
        # Render each kind of object
        obj_type = obj[0]

        def render_dish(img, num_onions, num_tomatoes):
            # Render the plate
            plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS["white"])

            # Coordinates for up to 3 items
            coords = [(0.65, 0.75), (0.75, 0.75), (0.85, 0.75)]
            
            # Render onions
            for i in range(num_onions):
                onion_fn = rendering.point_in_circle(coords[i][0], coords[i][1], 0.13)
                rendering.fill_coords(img, onion_fn, COLORS["orange"])

            # Render tomatoes
            for i in range(num_tomatoes):
                tomato_fn = rendering.point_in_circle(coords[i + num_onions][0], coords[i + num_onions][1], 0.13)
                rendering.fill_coords(img, tomato_fn, COLORS["red"])


        if obj_type == OBJECT_TO_INDEX['empty']:
            pass
        elif obj_type == OBJECT_TO_INDEX['onion']:
            onion_fn = rendering.point_in_circle(0.75, 0.75, 0.15)
            rendering.fill_coords(img, onion_fn, COLORS["yellow"])
        elif obj_type == OBJECT_TO_INDEX['tomato']:
            onion_fn = rendering.point_in_circle(0.75, 0.75, 0.15)
            rendering.fill_coords(img, onion_fn, COLORS["red"])
        elif obj_type == OBJECT_TO_INDEX['plate']:
            plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS["white"])
        elif obj_type == OBJECT_TO_INDEX['dish_0']:
            render_dish(img, num_onions=1, num_tomatoes=0)
        elif obj_type == OBJECT_TO_INDEX['dish_1']:
            render_dish(img, num_onions=2, num_tomatoes=0)
        elif obj_type == OBJECT_TO_INDEX['dish_2']:
            render_dish(img, num_onions=3, num_tomatoes=0)
        elif obj_type == OBJECT_TO_INDEX['dish_3']:
            render_dish(img, num_onions=0, num_tomatoes=1)
        elif obj_type == OBJECT_TO_INDEX['dish_4']:
            render_dish(img, num_onions=0, num_tomatoes=2)
        elif obj_type == OBJECT_TO_INDEX['dish_5']:
            render_dish(img, num_onions=0, num_tomatoes=3)
        elif obj_type == OBJECT_TO_INDEX['dish_6']:
            render_dish(img, num_onions=1, num_tomatoes=1)
        elif obj_type == OBJECT_TO_INDEX['dish_7']:
            render_dish(img, num_onions=2, num_tomatoes=1)
        elif obj_type == OBJECT_TO_INDEX['dish_8']:
            render_dish(img, num_onions=1, num_tomatoes=2)
        else:
            raise ValueError(f'Rendering object at index {obj_type} is currently unsupported.')

    @classmethod
    def _render_tile(
        cls,
        obj,
        highlight=False,
        agent_dir_idx=None,
        agent_inv=None,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """
        # Hash map lookup key for the cache
        if obj is not None and obj[0] == OBJECT_TO_INDEX['agent']:
            # Get inventory of this specific agent
            if agent_inv is not None:
                color_idx = obj[1]
                agent_inv = agent_inv[COLOR_TO_AGENT_INDEX[color_idx]]
                agent_inv = np.array([agent_inv, -1, 0])

            if agent_dir_idx is not None:
                obj = np.array(obj)

                # TODO: Fix this for multiagents. Currently the orientation of other agents is wrong
                if len(agent_dir_idx) == 1:
                    # Hacky way of making agent views orientations consistent with global view
                    obj[-1] = agent_dir_idx[0]

        no_object = obj is None or (
            obj[0] in [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['unseen']] \
            and obj[2] == 0
        )

        if not no_object:
            if agent_inv is not None and obj[0] == OBJECT_TO_INDEX['agent']:
                key = (*obj, *agent_inv, highlight, tile_size)
            else:
                key = (*obj, highlight, tile_size)
        else:
            key = (obj, highlight, tile_size)

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        rendering.fill_coords(img, rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if not no_object:
            OvercookedVisualizer._render_obj(obj, img)
            # render inventory
            if agent_inv is not None and obj[0] == OBJECT_TO_INDEX['agent']:
                OvercookedVisualizer._render_inv(agent_inv, img)


        if highlight:
            rendering.highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    @classmethod
    def _render_grid(
        cls,
        grid,
        tile_size=TILE_PIXELS,
        highlight_mask=None,
        agent_dir_idx=None,
        agent_inv=None):
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=grid.shape[:2], dtype=bool)

        # Compute the total grid size in pixels
        width_px = grid.shape[1]*tile_size
        height_px = grid.shape[0]*tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):		
                obj = grid[y,x,:]
                if obj[0] in [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['unseen']] \
                    and obj[2] == 0:
                    obj = None

                tile_img = OvercookedVisualizer._render_tile(
                    obj,
                    highlight=highlight_mask[y, x],
                    tile_size=tile_size,
                    agent_dir_idx=agent_dir_idx,
                    agent_inv=agent_inv,
                )

                ymin = y*tile_size
                ymax = (y+1)*tile_size
                xmin = x*tile_size
                xmax = (x+1)*tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def close(self):
        self.window.close()