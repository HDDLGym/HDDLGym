// Constants.js or at the top of your main game file
const ANIMATION_DURATION = 100;

const DIRECTION_TO_NAME = {
    '0,-1': 'NORTH',
    '0,1': 'SOUTH',
    '1,0': 'EAST',
    '-1,0': 'WEST'
};

const scene_config = {
    player_colors : {0: 'blue', 1: 'green'},
    tileSize : 80,
    animation_duration : ANIMATION_DURATION,
    show_post_cook_time : false,
    cook_time : 60,
    assets_loc : "./static/assets/",
    hud_size : 40
};

const onion_value = 2
const tomato_value = 1
const zoom_level = 2

////////////////

var game_config = {
    type: Phaser.WEBGL,
    antialias: false,
    roundPixels: true,
    pixelArt: true,
    audio: {
        noAudio: true
    },
};

////////////////


var graphics;

// Invoked at every state_pong event from server
// function drawState(state) {
//     if (graphics && graphics.isSceneInitialized()) {
//         graphics._drawState(state);
//     } else {
//         console.log("Scene is not yet initialized, waiting to set state");
//     }
// };
function drawState(state) {
    // Try catch necessary because state pongs can arrive before graphics manager has finished initializing
    try {
        graphics.set_state(state);

        if (state.message1 && state.message1_location) {
            const scene = graphics.game.scene.getScene('MainGameScene');
            scene.displayMessageAtLocation(state.message1, state.message1_location);
        }

        if (state.message2 && state.message2_location) {
            const scene = graphics.game.scene.getScene('MainGameScene');
            scene.displayMessageAtLocation2(state.message2, state.message2_location);
        }

    } catch {
        console.log("error updating state");
    }
};

// Invoked at 'start_game' event
function graphics_start(graphics_config) {
    graphics = new GraphicsManager(game_config, scene_config, graphics_config);
};

// Invoked at 'end_game' event
function graphics_end() {
    if (graphics && graphics.game.scene.isActive('MainGameScene')) {
        graphics.game.scene.getScene('MainGameScene').shutdown();
    }
    if (graphics) {
        graphics.game.renderer.destroy();
        graphics.game.loop.stop();
        graphics.game.destroy();
        graphics = null; // Clear the global variable
    }
}


class GraphicsManager {
    constructor(game_config, scene_config, graphics_config) {
        this.graphics_config = graphics_config;
        this.scene_config = scene_config;
        this.game_config = game_config;
        this.initGame();
    }

    initGame() {
        // Update scene_config with the start_info from graphics_config
        this.scene_config.terrain = this.graphics_config.start_info.terrain;
        this.scene_config.start_state = this.graphics_config.start_info.state;
        this.scene_config.currLayout = this.graphics_config.currLayout;

        // Update game_config with scene dimensions and parent container
        this.game_config.width =  (this.scene_config.tileSize * this.scene_config.terrain[0].length) * zoom_level;
        this.game_config.height = (this.scene_config.tileSize * this.scene_config.terrain.length + this.scene_config.hud_size) * zoom_level;
        this.game_config.parent = this.graphics_config.container_id;

        // Initialize the Phaser game with the updated game configuration
        this.game = new Phaser.Game(this.game_config);

        // Add the 'OvercookedScene' to the Phaser game. Use the key 'PlayGame' for the scene
        this.game.scene.add('MainGameScene', new OvercookedScene(this.scene_config), true);
    }

    isSceneInitialized() {
        let scene = this.game.scene.getScene('MainGameScene');
        return scene && scene.isInitialized;
    }

    set_state(state) {
        this.game.scene.getScene('MainGameScene').set_state(state);
    }
}

class OvercookedScene extends Phaser.Scene {
    constructor(config) {
        super({ key: "MainGameScene" });
        this.config = config;
        // this.sprites = { chefs: {}, objects: {} };
        this.sprites = {
            chefs: {},
            objects: {},
            allOrders: {},
            bonusOrders: {}
        };
        this.state = config.start_state.state;
        this.terrain = config.terrain;   //     
        this.tileSize = config.tileSize;   //     
        this.previousState = config.start_state;
        this.number = 0
        this.players = this.state ? this.state.players : [];        
        this.chefGroup = null;
        this.objectGroup = null;
        this.hudGroup = null;
        this.terrainGroup = null;
        this.bonusOrderGroup = null;
        this.allOrderGroup = null;       
        this.markedTilesGroup = null;
        this.currLayout = config.currLayout       
        this.labelsVisible = true;
        this.messageText = null; // Initialize message text variable

        this.hud_data = {
            score : config.start_state.score,
            time : config.start_state.time_left,
            bonus_orders : config.start_state.state.bonus_orders,
            all_orders : config.start_state.state.all_orders
        }

        this.prevBonusOrders = null;
        this.prevAllOrders = null;
        this.previousScore = null;
        this.previousTimeLeft = null;        
    }

    preload() {
        this.load.atlas("tiles", this.config.assets_loc + "terrain.png", this.config.assets_loc + "terrain.json");
        this.load.atlas("chefs", this.config.assets_loc + "chefs.png", this.config.assets_loc + "chefs.json");
        this.load.atlas("objects", this.config.assets_loc + "objects.png", this.config.assets_loc + "objects.json");
        this.load.multiatlas("soups", this.config.assets_loc + "soups.json", this.config.assets_loc);
        this.load.multiatlas("soups_key", this.config.assets_loc + "soups_key.json", this.config.assets_loc);
        this.load.image("serving_symbol", this.config.assets_loc + "icon_serving.png");
        this.load.image("coin", this.config.assets_loc + "coin.png");
        this.load.image("clock", this.config.assets_loc + "clock.png");
    }

    create() {
        this.cameras.main.setZoom(zoom_level);
        const worldCenterX = (this.game.config.width / 2) / zoom_level;
        const worldCenterY = (this.game.config.height / 2) / zoom_level - this.config.hud_size;
        this.cameras.main.centerOn(worldCenterX, worldCenterY);

        this.chefGroup = this.add.group();
        this.objectGroup = this.add.group();
        this.hudGroup = this.add.group();
        this.terrainGroup = this.add.group();
        this.bonusOrderGroup = this.add.group();
        this.allOrderGroup = this.add.group();
        this.markedTilesGroup = this.add.group();        

        this.chefSpritePool = new SpritePool(this, 'chefs', 20);
        this.ingredientSpritePool = new SpritePool(this, 'objects', 50);
        this.chefSpritePool.populatePool();
        this.ingredientSpritePool.populatePool();    
        this.createLevel();

        if (this.currLayout === 'experiment/tutorial_custom_0') {
            this.createLabels(0);
        } else if (this.currLayout === 'experiment/tutorial_custom_1_3recipes') {
            this.createLabels(1);
        } else if (this.currLayout === 'experiment/tutorial_custom_2') {
            this.createLabels(2);
        } 

        // this.createLabels(1);
        // this.createHUD();
        // this._drawState(this.state)
        this.isInitialized = true;        

        window.addEventListener('toggleLabels', () => {
            this.toggleLabels();
        });

        // this.input.keyboard.on('keydown-T', this.toggleLabels, this);

        // this.time.addEvent({
        //     delay: 1000, // 1000 milliseconds = 1 second
        //     callback: this.updateCookingTicks.bind(this), // Binding 'this' to the class instance
        //     loop: true
        // });     
    }

    /////////////////////////////////////////////////////////////////
    displayMessageAtLocation(message, location) {
        const tileSize = this.tileSize;
        const x = location[0] * tileSize - 60;
        const y = location[1] * tileSize - 23;

        if (!this.messageText) {
            this.messageText = this.add.text(x, y, message, {
                font: "14px Arial",
                fill: "#ffffff",
                backgroundColor: "rgba(0, 0, 0, 0.5)",
                padding: { x: 4, y: 4 },
                borderRadius: 3,
            });
        } else {
            this.messageText.setText(message);
            this.messageText.setPosition(x, y);
            this.messageText.setVisible(true);
        }

        // Hide the message after a certain time
        this.time.delayedCall(3000, () => {
            this.messageText.setVisible(false);
        });
    }

    hideMessage() {
        if (this.messageText) {
            this.messageText.setVisible(false);
        }
    }


    displayMessageAtLocation2(messageReward, location) {
        const tileSize = this.tileSize;
        // const x = location[0] * tileSize + 25 + 5;
        // const y = location[1] * tileSize + 25 + 5;
        const x = location[0] * tileSize + 5;
        const y = location[1] * tileSize + 5;
        const x_coin = x - 5;
        const y_coin = y - 5;
    
        // Convert messageReward to a numerical score
        const score = Number(messageReward);
        const textColor = score === 0 ? "red" : "green";
    
        if (!this.messageRewardText) {
            this.coin = this.add.sprite(x_coin, y_coin, "coin");
            this.coin.setDisplaySize(35, 35);
            this.coin.setOrigin(0, 0);
    
            this.messageRewardText = this.add.text(x, y, messageReward, {
                font: "20px Arial",
                fill: textColor,
            });
        } else {
            this.coin.setPosition(x_coin, y_coin);
            this.coin.setDisplaySize(35, 35);
            this.coin.setVisible(true);
    
            this.messageRewardText.setText(messageReward);
            this.messageRewardText.setPosition(x, y);
            this.messageRewardText.setStyle({ fill: textColor });
            this.messageRewardText.setVisible(true);
        }
    
        // Hide the messageReward and the coin after a certain time
        this.time.delayedCall(2000, () => {
            this.messageRewardText.setVisible(false);
            this.coin.setVisible(false);
        });
    }
    
    hideMessageReward2() {
        if (this.messageRewardText) {
            this.messageRewardText.setVisible(false);
        }
        if (this.coin) {
            this.coin.setVisible(false);
        }
    }


    /////////////////////////////////////////////////////////////////

    set_state(state) {
        this.hud_data.score = state.score;
        //this.hud_data.time = Math.round(state.time_left);
        this.hud_data.time = Math.round(state.time_left * 10) / 10;
        this.hud_data.bonus_orders = state.state.bonus_orders;
        this.hud_data.all_orders = state.state.all_orders;
        this.state = state.state;
    }

    update(time, delta)  {

        let deltaTimeInSeconds = delta / 1000;

        if (typeof(this.state) !== 'undefined') {
            this._drawState(this.state, this.sprites);
        }
        if (typeof(this.hud_data) !== 'undefined') {
            let { width, height } = this.game.canvas;
            let board_height = height / zoom_level - this.config.hud_size;
            this._drawHUD(this.hud_data, this.sprites, board_height);
        }        
    }

    // updateCookingTicks() {
    //     console.log('updateCookingTicks called');
    //     console.log('Objects:', this.state.objects);        
    //     for (let key in this.state.objects) {
    //         let obj = this.state.objects[key];
    //         if (obj.name === 'soup' && obj.hasOwnProperty('_cooking_tick') && obj._cooking_tick > 0) {
    //             obj._cooking_tick -= 1;
    //         }
    //     }
    // }

    shutdown() {
        this.tweens.killAll();

        this.chefGroup.destroy(true);
        this.objectGroup.destroy(true);
        this.hudGroup.destroy(true);
        this.terrainGroup.destroy(true);
        this.bonusOrderGroup.destroy(true);
        this.allOrderGroup.destroy(true);

        this.chefSpritePool.destroy();
        this.ingredientSpritePool.destroy();    
        this.markedTilesGroup.clear(true);

        // Clear any other references that might prevent garbage collection
        this.chefGroup = null;
        this.objectGroup = null;
        this.hudGroup = null;
        this.terrainGroup = null;
        this.bonusOrderGroup = null;
        this.allOrderGroup = null;        
    }


    createLabels(layoutType) {
        this.objectLabels = {};
        let labelPositions;
        if (layoutType === 0) {
            labelPositions = {
                'Pot': { x: 185, y: 5 }, // 150 , 40, 200
                'Onions': { x:8, y: 223 },
                'Empty\nCounter': { x: 403, y: 335 },
                'Your Partner': { x: 305, y: 50 },
                'Serving\nArea': { x: 485, y: 187 },
                'Bowls': { x: 255, y: 155 },
                'Soup Ingredients (2 onions)': { x: 195, y: 440 },
                'Points': { x: 195, y: 475 },
            };
        } else if (layoutType === 1) {
            labelPositions = {
                'Pot': { x: 185, y: 5 }, // 150 , 40, 200
                'Onions': { x:8, y: 223 },
                'Tomatoes': { x: 75, y: 150 },
                'Empty\nCounter': { x: 403, y: 335 },
                'Your Partner': { x: 305, y: 50 },
                'Serving\nArea': { x: 485, y: 185 },
                'Bowls': { x: 255, y: 155 },
                // 'Soup Ingredients (1 onion, 2 tomatoes)': { x: 190, y: 495 },
                // 'Points': { x: 190, y: 525 },
            };
        } else if (layoutType === 2) {
            labelPositions = {
                'Pot': { x: 185, y: 5 }, // 150 , 40, 200
                'Onions': { x:8, y: 223 },
                'Empty\nCounter': { x: 403, y: 170 },
                'Serving\nArea': { x: 485, y: 190 },
                'Bowls': { x: 255, y: 5 },
                // 'Soup Ingredients (1 onion, 2 tomatoes)': { x: 190, y: 495 },
                // 'Points': { x: 190, y: 525 },
            };
        }        

        for (const [key, position] of Object.entries(labelPositions)) {
            // Create a container for each label and its outline
            const labelContainer = this.add.container(0, 0);

            // Create the text label
            const label = this.add.text(position.x, position.y, key, {
                font: '20px Arial',
                fill: '#fff',
                align: 'center',
                padding: { x: 0, y: 0 },
            });

            // Create a red box (outline) around the label
            const graphics = this.add.graphics();
            graphics.lineStyle(1.5, 0xff0000, 1.0);
            graphics.strokeRect(label.x - 3, label.y - 2, label.width + 6, label.height + 4);

            // Add both the label and its outline to the container
            labelContainer.add(graphics);
            labelContainer.add(label);

            // Set the container's visibility based on this.labelsVisible
            labelContainer.setVisible(this.labelsVisible);

            // Store the container in the objectLabels object for easy access later
            this.objectLabels[key] = labelContainer;
        }
    }

    toggleLabels() {
        this.labelsVisible = !this.labelsVisible; // Toggle the visibility state
        Object.values(this.objectLabels).forEach(labelContainer => {
            labelContainer.setVisible(this.labelsVisible); // Update visibility for each container
        });
    }

    _drawState(state) {
        state.players.forEach((player, pi) => {
            let [x, y] = player.position;
            let dir = DIRECTION_TO_NAME[player.orientation].toLowerCase();
            let held_obj = player.held_object ? `_${player.held_object.name}` : "";
    
            if (player.held_object && player.held_object.name === 'soup') {
                let ingredients = player.held_object._ingredients.map(ing => ing.name);
                held_obj = ingredients.includes('onion') ? "_soup_onion" : "_soup_tomato";
            }
    
            let chefSpriteName = `chef${pi + 1}_${dir}${held_obj}.png`;
    
            if (!this.sprites.chefs[pi]) {
                let chefsprite = this.add.sprite(
                    Math.floor(this.tileSize* x),
                    Math.floor(this.tileSize * y),
                    "chefs",
                    chefSpriteName
                ).setDisplaySize(this.tileSize, this.tileSize).setDepth(1).setOrigin(0);
    
                this.sprites.chefs[pi] = { chefsprite };
            } else {
                let chefsprite = this.sprites.chefs[pi].chefsprite;
                chefsprite.setFrame(chefSpriteName);
    
                this.tweens.add({
                    targets: [chefsprite],
                    x: Math.floor(this.tileSize * x),
                    y: Math.floor(this.tileSize * y),
                    duration: this.config.animation_duration,
                    ease: 'Linear'
                });
            }
        });
    
        if (this._objectsStateChanged(state.objects, this.objects)) {
            this.state = state;
            this._drawEnvironmentObjects(this.state);
        }
    
        this.previousState = state;
    }


    createLevel() {
        this.cameras.main.setBackgroundColor('#e6b453');
        // this.cameras.main.setBackgroundColor('#986827');
        let terrain_to_img = {
            ' ': 'floor.png',
            'X': 'counter.png',
            'P': 'pot.png',
            'O': 'onions.png',
            'T': 'tomatoes.png',
            'D': 'dishes.png',
            'S': 'serve.png'
        };
        this.config.terrain.forEach((row, y) => {
            row.forEach((tileType, x) => {
                // let tile = this.add.sprite(
                //     this.config.tileSize * x,
                //     this.config.tileSize * y,
                //     "tiles",
                //     terrain_to_img[tileType]
                // );
                let tile = this.add.sprite(
                    this.config.tileSize * x,
                    this.config.tileSize * y,
                    "tiles",
                    terrain_to_img[tileType]
                );
                tile.setDisplaySize(this.config.tileSize, this.config.tileSize);
                tile.setOrigin(0);
                this.terrainGroup.add(tile);

                if (tileType === 'S') {
                    // Add the custom serving symbol on top of the tile
                    let symbol = this.add.sprite(
                        this.config.tileSize * x + this.config.tileSize / 2,
                        this.config.tileSize * y + this.config.tileSize / 2,
                        'serving_symbol'
                    );
                    symbol.setDisplaySize(this.config.tileSize * 0.7, this.config.tileSize * 0.7);
                    // symbol.setScale(0.2)
                    symbol.setOrigin(0.5, 0.5);
                    this.terrainGroup.add(symbol);
                }
            });
        });

        // Mark reachable counter tiles
        this.markableTiles = this.getMarkableTiles();
        for (let tile of this.markableTiles) {
            let [row, col] = tile.split(',').map(Number);
            this.markTile(row, col);
        }        
    }

    getMarkableTiles() {
        let markableTilesSet = new Set();
        for (let row = 0; row < this.terrain.length; row++) {
            for (let col = 0; col < this.terrain[row].length; col++) {
                if (this.terrain[row][col] === 'X' && this.isAdjacentToFloor(row, col)) {
                    markableTilesSet.add(`${row},${col}`);
                }
            }
        }
        return markableTilesSet;
    }

    isAdjacentToFloor(row, col) {
        const directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]; // Adjacent directions (up, right, down, left)
        for (let [dx, dy] of directions) {
            let newRow = row + dx, newCol = col + dy;
            if (newRow >= 0 && newRow < this.terrain.length && newCol >= 0 && newCol < this.terrain[row].length) {
                if (this.terrain[newRow][newCol] === ' ') {
                    return true;
                }
            }
        }
        return false;
    }

    markTile(row, col) {
        const style = {
            border: { thickness: 1.5, color: 0x5C4033, alpha: 0.8 },
            highlight: { color: 0xAA8844, alpha: 0.9 }
        };

        const { centerX, centerY, size } = this.calculateTileDimensions(row, col, style.border.thickness);
        const borderSides = this.calculateBorderSides(row, col);

        let highlight = this.createTileHighlight(centerX, centerY, size, style.highlight);
        this.markedTilesGroup.add(highlight);

        let borders = this.createTileBorder(centerX, centerY, size, style.border, borderSides);
        borders.forEach(border => this.markedTilesGroup.add(border));

        return this;
    }

    calculateTileDimensions(row, col, borderThickness) {
        return {
            centerX: this.tileSize * col + this.tileSize / 2,
            centerY: this.tileSize * row + this.tileSize / 2,
            size: this.tileSize - borderThickness
        };
    }

    createTileHighlight(x, y, size, { color, alpha }) {
        let highlight = this.add.rectangle(x, y, size, size, color)
            .setOrigin(0.5, 0.5)
            .setAlpha(alpha);
        return highlight; // Return the created object
    }

    calculateBorderSides(row, col) {
        let borders = { top: true, right: true, bottom: true, left: true };

        const isAdjacentTileMarked = (r, c) => this.markableTiles.has(`${r},${c}`);

        if (this.markableTiles.has(`${row},${col}`)) {
            if (row > 0 && isAdjacentTileMarked(row - 1, col)) borders.top = false;
            if (col < this.terrain[0].length - 1 && isAdjacentTileMarked(row, col + 1)) borders.right = false;
            if (row < this.terrain.length - 1 && isAdjacentTileMarked(row + 1, col)) borders.bottom = false;
            if (col > 0 && isAdjacentTileMarked(row, col - 1)) borders.left = false;
        }

        return borders;
    }

    createTileBorder(x, y, size, { thickness, color, alpha }, borderSides) {
        let halfSize = size / 2;
        let borderOffset = thickness / 2;
        let borders = [];

        if (borderSides.top) {
            let topBorder = this.add.line(0, 0, x, y - halfSize + borderOffset, x + 2 * halfSize + 2 * borderOffset, y - halfSize + borderOffset, color, alpha).setLineWidth(thickness);
            borders.push(topBorder);
        }
        if (borderSides.right) {
            let rightBorder = this.add.line(0, 0, x + halfSize - borderOffset, y + 2 * halfSize + 2 * borderOffset, x + halfSize - borderOffset, y, color, alpha).setLineWidth(thickness);
            borders.push(rightBorder);
        }
        if (borderSides.bottom) {
            let bottomBorder = this.add.line(0, 0, x, y + halfSize - borderOffset, x + 2 * halfSize + 2 * borderOffset, y + halfSize - borderOffset, color, alpha).setLineWidth(thickness);
            borders.push(bottomBorder);
        }
        if (borderSides.left) {
            let leftBorder = this.add.line(0, 0, x - halfSize + borderOffset, y + 2 * halfSize + 2 * borderOffset, x - halfSize + borderOffset, y, color, alpha).setLineWidth(thickness);
            borders.push(leftBorder);
        }

        return borders; // Return the array of created border objects
    } 

    createHUD() {
        this.currentScore = 0;
        this.currentTimeLeft = 120;

        // Score Display
        this.scoreText = this.add.text(16, 16, 'Score: 0', { fontSize: '32px', fill: '#fff' });
        this.scoreText.setScrollFactor(0); // Keeps the text fixed in the view

        // Time Left Display
        this.timeLeftText = this.add.text(16, 50, 'Time Left: 120', { fontSize: '32px', fill: '#fff' });
        this.timeLeftText.setScrollFactor(0);

        this.hudGroup.add(this.scoreText);
        this.hudGroup.add(this.timeLeftText);        
    }


    // Example score calculation function
    calculateScore(ingredients) {
        return ingredients.reduce((total, ingredient) => {
            return total + (ingredient === 'onion' ? onion_value : (ingredient === 'tomato' ? tomato_value : 0));
        }, 0);
    }


    _drawScore(score, sprites, board_height) {      
      
        if (this.previousScore !== score) {
            
            this.previousScore = score;        
            board_height -= 110;
            // score = "Overall Score:   "+ score;

            if (parseFloat(score) < 10) {
                score = " " + score;
            } else {
                score = "" + score;
            }

            if (typeof(sprites['score']) !== 'undefined') {
                sprites['score'].setText(score);
            }
            else {
                let coin = this.add.sprite(-3, board_height + 60 - 10, "coin");
                coin.setDisplaySize(60, 60);
                coin.setOrigin(-0.18, 0.1);
                coin.setDepth(0)

               
                // Update the x-coordinate in this.add.text(...)
                sprites['score'] = this.add.text(
                    20, board_height + 60 - 4, score,
                    {
                        font: "32px Arial",
                        fill: "black",
                        align: "left"
                    }
                )             
            }
        }
    }

    _drawTimeLeft(time_left, sprites, board_height) {
        if (this.previousTimeLeft !== time_left) {
            this.previousTimeLeft = time_left;        
            board_height -= 110;
            time_left = time_left;
            time_left = Math.round(time_left);

            let timeColor = time_left <= 30 ? "red" : (time_left <= 60 ? "yellow" : "green");
            let textStyle = {
                font: "32px Arial",
                fill: "black", 
                align: "left",
            };
            let textStyleIcon = {
                font: "47px Arial",
                fill: "black", 
                align: "left",
            };

            let x_position;
            if (time_left < 10) {
                x_position = this.game.canvas.width / zoom_level - 57 + 7;
            } else {
                x_position = this.game.canvas.width / zoom_level - 57;
            }

            if (typeof(sprites['time_left']) !== 'undefined') {
                sprites['time_left'].x = x_position;
                sprites['time_left'].setText(time_left);
                sprites['time_left'].setStyle(textStyle);
            } else {
                let clock = this.add.sprite(this.game.canvas.width / zoom_level - 70 - 10, board_height + 50, "clock");                
                clock.setDisplaySize(70, 70);
                // clock.setOrigin(-0.18, 0.1); # for 60 x 60
                clock.setOrigin(-0.08, 0.2);
                clock.setDepth(0)                
                // sprites['time_left_icon'] = this.add.text(this.game.canvas.width / zoom_level - 70, board_height + 50, "⏲️", textStyleIcon);

                
                if (this.currLayout === 'experiment/tutorial_custom_0') {
                        // time_left = '∞';
                        time_left = '0';
                    }
                if (this.currLayout === 'experiment/tutorial_custom_1_3recipes') {
                        time_left = '0';
                    }
                        
                sprites['time_left'] = this.add.text(x_position, board_height + 60 - 6, time_left, textStyle);
            }
        }
    }

    _playersStateChanged(newPlayers, oldPlayers) {
        return JSON.stringify(newPlayers) !== JSON.stringify(oldPlayers);
    }

    _objectsStateChanged(newObjects, oldObjects) {
        return JSON.stringify(newObjects) !== JSON.stringify(oldObjects);
    }

    _drawHUD(hud_data, sprites, board_height) {
        if (typeof(hud_data.time) !== 'undefined') {
            this._drawTimeLeft(hud_data.time, sprites, board_height);
        }
        if (typeof(hud_data.score) !== 'undefined') {
            this._drawScore(hud_data.score, sprites, board_height);
        }
        if (typeof(hud_data.all_orders) !== 'undefined') {
            this._drawAllOrders(hud_data.all_orders, sprites, board_height);
        }
        // if (typeof(hud_data.bonus_orders) !== 'undefined') {
        //     this._drawBonusOrders(hud_data.bonus_orders, sprites, board_height);
        // }
    }

    _drawBonusOrders(orders, sprites, board_height) {
        board_height += 42;
        if (Array.isArray(orders)) {
            const currentOrders = JSON.stringify(orders);
            if (this.prevBonusOrders === currentOrders) {
                return;
            }
            this.prevBonusOrders = currentOrders;

            let orders_str = "Bonus Orders:";
            let textStyle = {
                font: "20px Arial",
                fill: "black", 
                align: "left",
                padding: 5,
            };
            let pointStyle = {
                font: "14px Arial",
                fill: "black", 
                align: "left",
                padding: 5,
            };            

            if (!sprites['bonus_orders']) {
                sprites['bonus_orders'] = { 'orders': [] };
            }

            sprites['bonus_orders']['orders'].forEach(element => element.destroy());
            sprites['bonus_orders']['orders'] = [];

            if (sprites['bonus_orders']['str']) {
                sprites['bonus_orders']['str'].setText(orders_str);
            } else {
                sprites['bonus_orders']['str'] = this.add.text(5, board_height + 60, orders_str, textStyle);
                sprites['bonus_orders']['str'] = this.add.text(5, board_height + 85, 'Soup Scores:', pointStyle);
            }

            orders.forEach((order, i) => {
                let xPosition = 135 + 40 * i;
                let spriteFrame = this._ingredientsToSpriteFrame(order['ingredients'], "done");
                let orderSprite = this.add.sprite(xPosition, board_height + 40, "soups_key", spriteFrame);
                sprites['bonus_orders']['orders'].push(orderSprite);
                orderSprite.setDisplaySize(60, 60);
                orderSprite.setOrigin(0);
                orderSprite.depth = 1;

                let score = this.calculateScore_bonus(order['ingredients']);
                let scoreText = this.add.text(xPosition + 30, board_height + 88, score.toString(), {
                    font: "14px Arial",
                    fill: "black",
                    align: "center"
                }).setScrollFactor(0);
                scoreText.setOrigin(0.5, 0);
            });
        }
    }


    // Example score calculation function (you should implement this based on your logic)
    calculateScore_bonus(ingredients) {
        return ingredients.reduce((total, ingredient) => {
            if (ingredient === 'onion') {
                return total + onion_value;
            } else if (ingredient === 'tomato') {
                return total + tomato_value;
            } else {
                return total;
            }
        }, 0) * 2;
    }
    _drawAllOrders(orders, sprites, board_height) {
        board_height += 32;        
        if (Array.isArray(orders)) {
            // Filter out orders that are also in the bonus orders
            const bonusOrders = this.hud_data.bonus_orders.map(order => JSON.stringify(order.ingredients.sort()));

            // const uniqueOrders = orders.filter(order => !bonusOrders.includes(JSON.stringify(order.ingredients.sort())));
            // const currentOrders = JSON.stringify(uniqueOrders);

            const uniqueOrders = orders
            const currentOrders = JSON.stringify(orders);


            // const currentOrders = JSON.stringify(orders);

            if (this.prevAllOrders === currentOrders) {
                return;
            }
            this.prevAllOrders = currentOrders;``

            let orders_str = "Soup Menu:";
            let offsetx = 0;
            console.log('ok')
            console.log(this.terrain[0].length) // 4 = width
            console.log(this.terrain.length) // 6 = heights
            let offsety = -(this.tileSize * this.terrain.length + 77);


            let textStyle = {
                font: "20px Arial",
                fill: "black", 
                align: "left",
                padding: 5,
            };                               
            let pointStyle = {
                font: "20px Arial",
                fill: "black", 
                align: "left",
                padding: 5,
            };                               
            if (!sprites['all_orders']) {
                sprites['all_orders'] = { 'orders': [] };
            }

            sprites['all_orders']['orders'].forEach(element => element.destroy());
            sprites['all_orders']['orders'] = [];


            uniqueOrders.forEach((order, i) => {
                // let xPosition = 135 + 60 * i;
                let iconSize = 50
                // let xPosition = 30 + (iconSize + 10 + 70) * i;
                let xPosition = 25 + (iconSize + 10 + 50) * i;

                // Determine fill color based on whether the order is a bonus order
                let isBonusOrder = bonusOrders.includes(JSON.stringify(order.ingredients.sort()));
                // let fillColor = isBonusOrder ? 0xFF0000 : 0xFFFFFF;

                let borderProps = isBonusOrder ? { color: 0xFF0000, thickness: 4 } : { color: 0x000000, thickness: 2 };
                let borderColor = borderProps.color;
                let borderThickness = borderProps.thickness;


                // Draw filled rectangle first
                let graphics = this.add.graphics();
                graphics.fillStyle(0xFFFFFF, 1);
                // graphics.fillStyle(fillColor, 1);
                graphics.fillRect(xPosition - 1 - 15, board_height - 4 + 15 + offsety - 5, iconSize + 2 + 50, iconSize - 20 + 10);
                graphics.lineStyle(borderThickness, borderColor, 1);
                graphics.strokeRect(xPosition - 1 - 15, board_height - 4 + 15 + offsety - 5, iconSize + 2 + 50, iconSize - 20 + 10);
                
                let spriteFrame = this._ingredientsToSpriteFrame(order['ingredients'], "done");
                
                
                let orderSprite;
                if (order.ingredients.length === 1) {
                    let text = this.add.text(xPosition + 25 - 10, board_height + 15 + offsety, 'x 1', {
                        font: '16px Arial',
                        fill: '#000000'
                    });
                    orderSprite = this.add.sprite(xPosition - 10 - 10, 5 + board_height - 4 + offsety, "soups_key", spriteFrame);
                } else {
                    orderSprite = this.add.sprite(xPosition - 10, 5 + board_height - 4 + offsety, "soups_key", spriteFrame);
                }
                
                sprites['all_orders']['orders'].push(orderSprite);
                orderSprite.setDisplaySize(iconSize, iconSize);
                orderSprite.setOrigin(0);
                orderSprite.depth = 1;

                
                let score = this.calculateScore(order['ingredients']);
                if (isBonusOrder) {
                    score *= 2;
                }
                
                
                /////////////////////////////////////////////////////////////////////////////////
                let coin = this.add.sprite(xPosition + 60, board_height + 13 + offsety, "coin");
                coin.setDisplaySize(35, 35);
                coin.setOrigin(0.5, 0.14);
                /////////////////////////////////////////////////////////////////////////////////
                

                let scoreText = this.add.text(xPosition + 60, board_height + 13 + offsety, score.toString(), {
                    font: "22px Arial",
                    fill: "black",
                    align: "center"
                });
                scoreText.setOrigin(0.5, 0);
            });
        }
    }    

    _drawEnvironmentObjects(state) {
        let fps = 10

        // Clear previous environment objects
        for (let key in this.sprites.objects) {
            if (this.sprites.objects.hasOwnProperty(key)) {
                if (this.sprites.objects[key].timesprite) {
                    this.sprites.objects[key].timesprite.destroy();
                }
                this.sprites.objects[key].destroy();
                delete this.sprites.objects[key];
            }
        }
        this.sprites.objects = {};

        // Draw new environment objects
        for (let objpos in state.objects) {
            if (!state.objects.hasOwnProperty(objpos)) continue;
            let obj = state.objects[objpos];
            let [x, y] = obj.position;
            let terrainType = this.config.terrain[y][x];
            let spriteFrame;

            if (obj.name === 'soup' && terrainType === 'P') {
                // Handle soup in a pot
                let ingredients = obj._ingredients.map(ing => ing.name);
                let soupStatus = obj.is_ready ? "cooked" : "idle";
                spriteFrame = this._ingredientsToSpriteFrame(ingredients, soupStatus);

                let objsprite = this._createEnvironmentSprite(x, y, "soups", spriteFrame);

                // Show timer for cooking objects
                if (obj.hasOwnProperty('_cooking_tick') && obj._cooking_tick !== -1) {
                    let remainingTime = Math.max(0, obj.cook_time - obj._cooking_tick);
                    let scaledRemainingTime = remainingTime / fps; // shows every 0.1s as fps = 10                    
                    let positionKey = `${x},${y}`;
                    if (this.sprites.objects[positionKey] && this.sprites.objects[positionKey].timesprite) {
                        this.sprites.objects[positionKey].timesprite.setText(String(obj._cooking_tick));
                    } else {
                        let showTime = obj._cooking_tick <= obj.cook_time || this.show_post_cook_time;
                        let num;
                        if (remainingTime == 0) {
                            num = 0;
                        } else {
                            num = 8;
                        }                       
                        if (showTime) {
                            let timesprite = this.add.text(
                                this.config.tileSize * (x + 0.5) - 10 - num,
                                this.config.tileSize * (y + 0.6) - 10 + 13,
                                String(scaledRemainingTime),
                                {
                                    font: "25px Arial",
                                    fill: "red",
                                    align: "center"
                                }
                            );
                            timesprite.setDepth(2);
                            objsprite.timesprite = timesprite;
                        }
                    }

                    this.sprites.objects[positionKey] = objsprite;
                }        


            } else if (obj.name === 'soup') {
                // Soup not in a pot (e.g., on a plate or counter)
                let ingredients = obj._ingredients.map(ing => ing.name);
                spriteFrame = this._ingredientsToSpriteFrame(ingredients, "done");

                let objsprite = this._createEnvironmentSprite(x, y, "soups", spriteFrame);
                // this.sprites.objects[objpos] = { sprite: objsprite };
            } else {
                // Other objects
                spriteFrame = `${obj.name}.png`;
                let objsprite = this._createEnvironmentSprite(x, y, obj.name === 'soup' ? "soups" : "objects", spriteFrame);
                // this.sprites.objects[objpos] = { sprite: objsprite };
            }
        }
    }


    _createEnvironmentSprite(x, y, atlasKey, frameKey) {
        let sprite = this.add.sprite(this.config.tileSize * x, this.config.tileSize * y, atlasKey, frameKey);
        sprite.setDisplaySize(this.config.tileSize, this.config.tileSize);
        sprite.setOrigin(0);
        sprite.setDepth(1);
        let positionKey = `${x},${y}`;
        this.sprites.objects[positionKey] = sprite;
        return sprite;
    }

    _ingredientsToSpriteFrame(ingredients, status) {
        let num_tomatoes = ingredients.filter(x => x === 'tomato').length;
        let num_onions = ingredients.filter(x => x === 'onion').length;
        return `soup_${status}_tomato_${num_tomatoes}_onion_${num_onions}.png`
    }    
}

class SpritePool {
    constructor(scene, spriteKey, maxSize) {
        this.scene = scene;
        this.spriteKey = spriteKey;
        this.maxSize = maxSize;
        this.pool = [];
    }

    populatePool() {
        for (let i = 0; i < this.maxSize; i++) {
            let sprite = this.scene.add.sprite(0, 0, this.spriteKey).setVisible(false);
            this.pool.push(sprite);
        }
    }

    getSprite() {
        let sprite = this.pool.find(s => !s.visible);
        if (!sprite) {
            console.warn('Sprite pool exhausted. Consider increasing pool size.');
            sprite = this.scene.add.sprite(0, 0, this.spriteKey).setVisible(true);
            this.pool.push(sprite);
        }
        return sprite;
    }

    releaseSprite(sprite) {
        sprite.setVisible(false);
        sprite.x = 0;
        sprite.y = 0;
        sprite.clearTint();
        sprite.anims.stop();        
    }

    destroy() {
        this.pool.forEach(sprite => sprite.destroy());
        this.pool.length = 0;
    }    
}