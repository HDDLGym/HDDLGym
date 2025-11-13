import os
import sys

# Import and patch the production eventlet server if necessary
if os.getenv("FLASK_ENV", "production") == "production":
    import eventlet

    eventlet.monkey_patch()

import atexit
import json
import logging

# All other imports must come after patch to ensure eventlet compatibility
import pickle
import queue
from datetime import datetime
from threading import Lock, Timer

import game
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
# from flask_session import Session
from game import Game, OvercookedGame
from utils import ThreadSafeDict, ThreadSafeSet
import time
from flask_wtf import FlaskForm
from wtforms import BooleanField, SubmitField
import boto3
import gzip 

###############################################
# from load_RL_policy import network
###############################################


### Thoughts -- where I'll log potential issues/ideas as they come up
# Should make game driver code more error robust -- if overcooked randomlly errors we should catch it and report it to user
# Right now, if one user 'join's before other user's 'join' finishes, they won't end up in same game
# Could use a monitor on a conditional to block all global ops during calls to _ensure_consistent_state for debugging
# Could cap number of sinlge- and multi-player games separately since the latter has much higher RAM and CPU usage

###########
# Globals #
###########

# Read in global config
CONF_PATH = os.getenv("CONF_PATH", "config.json")
with open(CONF_PATH, "r") as f:
    CONFIG = json.load(f)

# Where errors will be logged
LOGFILE = CONFIG["logfile"]

# Available layout names
LAYOUTS = CONFIG["layouts"]
print(LAYOUTS)

# Values that are standard across layouts
LAYOUT_GLOBALS = CONFIG["layout_globals"]

# Maximum allowable game length (in seconds)
MAX_GAME_LENGTH = CONFIG["MAX_GAME_LENGTH"]

# Path to where pre-trained agents will be stored on server
AGENT_DIR = CONFIG["AGENT_DIR"]

# Maximum number of games that can run concurrently. Contrained by available memory and CPU
MAX_GAMES = CONFIG["MAX_GAMES"]

# Frames per second cap for serving to client
MAX_FPS = CONFIG["MAX_FPS"]

# Default configuration for predefined experiment
PREDEFINED_CONFIG = json.dumps(CONFIG["predefined"])

# Default configuration for tutorial
TUTORIAL_CONFIG = json.dumps(CONFIG["tutorial"])

# Global queue of available IDs. This is how we synch game creation and keep track of how many games are in memory
FREE_IDS = queue.Queue(maxsize=MAX_GAMES)

# Bitmap that indicates whether ID is currently in use. Game with ID=i is "freed" by setting FREE_MAP[i] = True
FREE_MAP = ThreadSafeDict()

# Initialize our ID tracking data
for i in range(MAX_GAMES):
    FREE_IDS.put(i)
    FREE_MAP[i] = True

# Mapping of game-id to game objects
GAMES = ThreadSafeDict()

# Set of games IDs that are currently being played
ACTIVE_GAMES = ThreadSafeSet()

# Queue of games IDs that are waiting for additional players to join. Note that some of these IDs might
# be stale (i.e. if FREE_MAP[id] = True)
WAITING_GAMES = queue.Queue()

# Mapping of users to locks associated with the ID. Enforces user-level serialization
USERS = ThreadSafeDict()
ACTIVE_SESSIONS = ThreadSafeDict()
ACTIVE_SESSIONS_UserTypes = ThreadSafeDict()
USER_SESSIONS = {}
LOG_USERS = ThreadSafeDict()
USERNAME_TIMERS = ThreadSafeDict()
USER_SCORES = ThreadSafeDict()


# Mapping of user id's to the current game (room) they are in
USER_ROOMS = ThreadSafeDict()

# Mapping of string game names to corresponding classes
GAME_NAME_TO_CLS = {
    "overcooked": OvercookedGame,
}

game._configure(MAX_GAME_LENGTH, AGENT_DIR)


#######################
# Flask Configuration #
#######################

# Create and configure flask app
app = Flask(__name__, template_folder=os.path.join("static", "templates"))
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
app.config["DEBUG"] = os.getenv("FLASK_ENV", "production") == "development"
# app.config['SESSION_TYPE'] = 'filesystem'
# Session(app)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=app.config["DEBUG"])

active_users = 0
max_users = 50
auth_needed = False
METRICS_ACCESS_PASSWORD = "hi"
USERNAME_TIMEOUT_SECONDS = 1000
save_feedback = False
save_user_info = False
collect_data = False
collect_data_local = False

# Attach handler for logging errors to file
handler = logging.FileHandler(LOGFILE)
handler.setLevel(logging.ERROR)
app.logger.addHandler(handler)

#################################
# Global Coordination Functions #
#################################


def try_create_game(game_name, **kwargs):
    """
    Tries to create a brand new Game object based on parameters in `kwargs`

    Returns (Game, Error) that represent a pointer to a game object, and error that occured
    during creation, if any. In case of error, `Game` returned in None. In case of sucess,
    `Error` returned is None

    Possible Errors:
        - Runtime error if server is at max game capacity
        - Propogate any error that occured in game __init__ function
    """
    try:
        curr_id = FREE_IDS.get(block=False)
        assert FREE_MAP[curr_id], "Current id is already in use"
        game_cls = GAME_NAME_TO_CLS.get(game_name, OvercookedGame)
        game = game_cls(id=curr_id, **kwargs)
    except queue.Empty:
        err = RuntimeError("Server at max capacity")
        return None, err
    except Exception as e:
        return None, e
    else:
        GAMES[game.id] = game
        FREE_MAP[game.id] = False
        return game, None


def cleanup_game(game: OvercookedGame):
    if FREE_MAP[game.id]:
        raise ValueError("Double free on a game")

    # User tracking
    for user_id in game.players:
        leave_curr_room(user_id)

    # Socketio tracking
    socketio.close_room(game.id)
    # Game tracking
    FREE_MAP[game.id] = True
    FREE_IDS.put(game.id)
    del GAMES[game.id]

    if game.id in ACTIVE_GAMES:
        ACTIVE_GAMES.remove(game.id)


def get_game(game_id):
    return GAMES.get(game_id, None)


def get_curr_game(user_id):
    return get_game(get_curr_room(user_id))


def get_curr_room(user_id):
    return USER_ROOMS.get(user_id, None)


def set_curr_room(user_id, room_id):
    USER_ROOMS[user_id] = room_id


def leave_curr_room(user_id):
    del USER_ROOMS[user_id]


def get_waiting_game():
    """
    Return a pointer to a waiting game, if one exists

    Note: The use of a queue ensures that no two threads will ever receive the same pointer, unless
    the waiting game's ID is re-added to the WAITING_GAMES queue
    """
    try:
        waiting_id = WAITING_GAMES.get(block=False)
        while FREE_MAP[waiting_id]:
            waiting_id = WAITING_GAMES.get(block=False)
    except queue.Empty:
        return None
    else:
        return get_game(waiting_id)


class TermsForm(FlaskForm):
    accept_all_terms = BooleanField('I have read and agree to all the terms and conditions.', default=True)
    submit = SubmitField('Accept')

##########################
# Socket Handler Helpers #
##########################


def _leave_game(user_id):
    """
    Removes `user_id` from it's current game, if it exists. Rebroadcast updated game state to all
    other users in the relevant game.

    Leaving an active game force-ends the game for all other users, if they exist

    Leaving a waiting game causes the garbage collection of game memory, if no other users are in the
    game after `user_id` is removed
    """
    # Get pointer to current game if it exists
    game = get_curr_game(user_id)

    if not game:
        # Cannot leave a game if not currently in one
        return False

    # Acquire this game's lock to ensure all global state updates are atomic
    with game.lock:
        # Update socket state maintained by socketio
        leave_room(game.id)

        # Update user data maintained by this app
        leave_curr_room(user_id)

        # Update game state maintained by game object
        if user_id in game.players:
            game.remove_player(user_id)
        else:
            game.remove_spectator(user_id)

        # Whether the game was active before the user left
        was_active = game.id in ACTIVE_GAMES

        # Rebroadcast data and handle cleanup based on the transition caused by leaving
        if was_active and game.is_empty():
            # Active -> Empty
            game.deactivate()
        elif game.is_empty():
            # Waiting -> Empty
            cleanup_game(game)
        elif not was_active:
            # Waiting -> Waiting
            emit("waiting", {"in_game": True}, room=game.id)
        elif was_active and game.is_ready():
            # Active -> Active
            pass
        elif was_active and not game.is_empty():
            # Active -> Waiting
            game.deactivate()

    return was_active


def _create_game(user_id, game_name, params={}, username_id=None, user_type=None):
    game, err = try_create_game(game_name, **params)
    if not game:
        emit("creation_failed", {"error": err.__repr__()})
        return
    spectating = True
    with game.lock:
        if not game.is_full():
            spectating = False
            game.add_player(user_id, username_id=username_id, user_type=user_type)
        else:
            spectating = True
            game.add_spectator(user_id)
        join_room(game.id)
        set_curr_room(user_id, game.id)
        if game.is_ready():
            game.activate()
            ACTIVE_GAMES.add(game.id)
            emit(
                "start_game",
                {"spectating": spectating, "start_info": game.to_json(), "currLayout": game.curr_layout},
                room=game.id,
            )
            socketio.start_background_task(play_game, game, fps=6)
        else:
            WAITING_GAMES.put(game.id)
            emit("waiting", {"in_game": True}, room=game.id)


#####################
# Debugging Helpers #
#####################


def _ensure_consistent_state():
    """
    Simple sanity checks of invariants on global state data

    Let ACTIVE be the set of all active game IDs, GAMES be the set of all existing
    game IDs, and WAITING be the set of all waiting (non-stale) game IDs. Note that
    a game could be in the WAITING_GAMES queue but no longer exist (indicated by
    the FREE_MAP)

    - Intersection of WAITING and ACTIVE games must be empty set
    - Union of WAITING and ACTIVE must be equal to GAMES
    - id \in FREE_IDS => FREE_MAP[id]
    - id \in ACTIVE_GAMES => Game in active state
    - id \in WAITING_GAMES => Game in inactive state
    """
    waiting_games = set()
    active_games = set()
    all_games = set(GAMES)

    for game_id in list(FREE_IDS.queue):
        assert FREE_MAP[game_id], "Freemap in inconsistent state"

    for game_id in list(WAITING_GAMES.queue):
        if not FREE_MAP[game_id]:
            waiting_games.add(game_id)

    for game_id in ACTIVE_GAMES:
        active_games.add(game_id)

    assert (
        waiting_games.union(active_games) == all_games
    ), "WAITING union ACTIVE != ALL"

    assert not waiting_games.intersection(
        active_games
    ), "WAITING intersect ACTIVE != EMPTY"

    assert all(
        [get_game(g_id)._is_active for g_id in active_games]
    ), "Active ID in waiting state"
    assert all(
        [not get_game(g_id)._id_active for g_id in waiting_games]
    ), "Waiting ID in active state"


def get_agent_names():
    return [
        d
        for d in os.listdir(AGENT_DIR)
        if os.path.isdir(os.path.join(AGENT_DIR, d))
    ]


######################
# Application routes #
######################


@app.route("/experiment")
def predefined():
    if 'username_id' not in session and auth_needed:
        return redirect(url_for('enter_details'))
    username_id = get_user_id()
    username_type = get_user_type()    
    log_event(username_id, username_type,"experiment","experiment Connected")                 
    uid = request.args.get("UID")

    num_layouts = len(CONFIG["predefined"]["experimentParams"]["layouts"])

    current_layout_number = 1

    return render_template(
        "experiment.html",
        uid=uid,
        config=PREDEFINED_CONFIG,
        num_layouts=num_layouts,
        username_id=username_id,
        layout_number=current_layout_number
    )



@app.route('/error_page')
def error_page():
    reason = request.args.get('reason', 'default_reason')
    titles = {
        'server_full': 'Server Capacity Reached',
        'multiple_browsers': 'Browser Session Limit',
    }
    messages = {
        'server_full': 'Sorry, the maximum number of allowed players are on the AWS instance. Please try again later.',
        'multiple_browsers': 'Sorry, only one browser is allowed per user.',
    }
    title = titles.get(reason, 'Experiment Ended')
    message = messages.get(reason, 'The experiment has ended.')
    return render_template('end_experiment.html', title=title, message=message)



@app.route('/end_experiment')
def end_experiment():
    
    username_id = session.get('username_id', None)
    USER_SESSIONS.pop(username_id, None)
    session.pop('username_id', None)    
    
    reason = request.args.get('reason', 'default_reason')
    titles = {
        'disconnection': 'Experiment Ended Due to Inactivity',
        'early_termination': 'Experiment Ended Due to Leaving',
        'finished-experiment': 'The Experiment has Finished',
    }
    messages = {
        'disconnection': 'Previous experiment ended due being disconnected for over 2 minutes.',
        'early_termination': 'You left the experiment and it has ended.',
        'finished-experiment': 'Thank you!',
    }
    title = titles.get(reason, 'Experiment Ended')
    message = messages.get(reason, 'The experiment has ended.')
    return render_template('end_experiment.html', title=title, message=message)



def datetime_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")



@app.route('/', methods=['GET', 'POST'])
def terms():
    if 'PROLIFIC_PID' in request.args and 'STUDY_ID' in request.args and 'SESSION_ID' in request.args:
        session['username_id'] = request.args.get('PROLIFIC_PID')
        session['study_id'] = request.args.get('STUDY_ID')
        session['session_id'] = request.args.get('SESSION_ID')
        session['user_type'] = 'prolific_'
    else:
        session['user_type'] = 'regular_'    
    
    if 'username_id' not in session and auth_needed:
        return redirect(url_for('enter_details'))
    form = TermsForm()
    username_id = get_user_id()
    username_type = get_user_type()
    log_event(username_id, username_type, "terms", "terms Connected")
    if request.method == 'POST':
        if form.validate_on_submit():
            return redirect(url_for('user_info'))            
    return render_template('terms.html', form=form, username_id=username_id)



@app.route("/metrics")
def metrics():
    # url/metrics?password=hi&collect_data=true
    global collect_data, auth_needed, save_feedback, save_user_info, collect_data_local
    
    collect_data_param = request.args.get('collect_data')
    if collect_data_param is not None:
        collect_data = collect_data_param.lower() == 'true'


    collect_data_local_param = request.args.get('collect_data_local')
    if collect_data_local_param is not None:
        collect_data_local = collect_data_local_param.lower() == 'true'

    save_feedback_param = request.args.get('save_feedback')
    if save_feedback_param is not None:
        save_feedback = save_feedback_param.lower() == 'true'

    save_user_info_param = request.args.get('save_user_info')
    if save_user_info_param is not None:
        save_user_info = save_user_info_param.lower() == 'true'

    auth_needed_param = request.args.get('auth_needed')
    if auth_needed_param is not None:
        auth_needed = auth_needed_param.lower() == 'true'

    password = request.args.get('password')
    new_experiment_type = request.args.get('experiment_type')

    if password != METRICS_ACCESS_PASSWORD:
        return jsonify({"error": "Unauthorized access"}), 401

    if new_experiment_type:
        try:
            new_experiment_type = int(new_experiment_type)
            game.set_experiment_type(new_experiment_type)
        except ValueError:
            return jsonify({"error": "Invalid experiment type"}), 400

    current_experiment_type = game.EXPERIMENT_TYPE
    
    user_request_ids = list(USERS.keys())
    active_prolific_ids = {str(ACTIVE_SESSIONS.get(uid, 'No ID')) : uid for uid in user_request_ids}
    active_user_type = {str(ACTIVE_SESSIONS_UserTypes.get(uid, 'No ID')) : uid for uid in user_request_ids}
    
    user_logs_data = {username_id: logs for username_id, logs in LOG_USERS.items()}    
    user_scores_data = {username_id: scores for username_id, scores in USER_SCORES.items()}
        
    # Include the available IDs in FREE_IDS and the state of FREE_MAP
    available_ids = list(FREE_IDS.queue)
    free_map_state = dict(FREE_MAP)

    resp = {
        "active_users": len(USERS),
        "prolific_ids_active": active_prolific_ids,        
        "active_user_type": active_user_type,        
        "user_logs": user_logs_data,        
        "user_scores": user_scores_data,        
        "auth_needed": auth_needed,        
        "collect_data": collect_data, 
        "collect_data_local": collect_data_local, 
        "save_feedback": save_feedback, 
        "save_user_info": save_user_info, 
        "experiment_type": current_experiment_type, 
        "available_ids": available_ids,
        "free_map_state": free_map_state,
        # "user_request_ids": user_request_ids,
    }
    return jsonify(resp)



#########################
# Socket Event Handlers #
#########################

# Asynchronous handling of client-side socket events. Note that the socket persists even after the
# event has been handled. This allows for more rapid data communication, as a handshake only has to
# happen once at the beginning. Thus, socket events are used for all game updates, where more rapid
# communication is needed

def get_user_id():
    username_id = session.get('username_id')
    return username_id

def get_user_type():
    user_type = session.get('user_type')
    return user_type


def log_event(username_id, username_type, event_type, event_description):
    timestamp = datetime.now()
    event = {"timestamp": timestamp, "event_type": event_type, "description": event_description}

    if username_id is None or username_id == 'Not provided': return

    username_id_type = username_type + username_id

    if username_id_type not in LOG_USERS:
        LOG_USERS[username_id_type] = []
    LOG_USERS[username_id_type].append(event)


def creation_params(params):
    """
    This function extracts the dataCollection and oldDynamics settings from the input and
    process them before sending them to game creation
    """
    # this params file should be a dictionary that can have these keys:
    # playerZero: human/Rllib*agent
    # playerOne: human/Rllib*agent
    # layout: one of the layouts in the config file, I don't think this one is used
    # gameTime: time in seconds
    # oldDynamics: on/off
    # dataCollection: on/off
    # layouts: [layout in the config file], this one determines which layout to use, and if there is more than one layout, a series of game is run back to back
    #

    use_old = False
    if "oldDynamics" in params and params["oldDynamics"] == "on":
        params["mdp_params"] = {"old_dynamics": True}
        use_old = True

    if "dataCollection" in params and params["dataCollection"] == "on":
        # config the necessary setting to properly save data
        params["dataCollection"] = True
        mapping = {"human": "H"}
        # gameType is either HH, HA, AH, AA depending on the config
        gameType = "{}{}".format(
            mapping.get(params["playerZero"], "A"),
            mapping.get(params["playerOne"], "A"),
        )
        params["collection_config"] = {
            "time": datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
            "type": gameType,
        }
        if use_old:
            params["collection_config"]["old_dynamics"] = "Old"
        else:
            params["collection_config"]["old_dynamics"] = "New"

    else:
        params["dataCollection"] = False


@socketio.on("create")
def on_create(data):
    user_id = request.sid
    username_id = ACTIVE_SESSIONS.get(user_id)    
    with USERS[user_id]:
        # Retrieve current game if one exists
        curr_game = get_curr_game(user_id)
        if curr_game:
            # Cannot create if currently in a game
            return

        params = data.get("params", {})

        creation_params(params)

        game_name = data.get("game_name", "overcooked")
        _create_game(user_id, game_name, params, username_id=username_id)


@socketio.on("join")
def on_join(data):
    user_id = request.sid
    username_id = ACTIVE_SESSIONS.get(user_id)
    user_type = ACTIVE_SESSIONS_UserTypes.get(user_id)
    with USERS[user_id]:
        create_if_not_found = data.get("create_if_not_found", True)

        # Retrieve current game if one exists
        curr_game = get_curr_game(user_id)
        if curr_game:
            # Cannot join if currently in a game
            return

        # Retrieve a currently open game if one exists
        
        if data['game_name'] == "tutorial":
            params = data.get("params", {})
            creation_params(params)
            game_name = data.get("game_name", "tutorial")
            _create_game(user_id, game_name, params, username_id=username_id, user_type=user_type)
            return           

        game = get_waiting_game()

        if not game and create_if_not_found:
            # No available game was found so create a game
            params = data.get("params", {})
            creation_params(params)
            game_name = data.get("game_name", "overcooked")
            _create_game(user_id, game_name, params, username_id=username_id, user_type=user_type)
            return

        elif not game:
            # No available game was found so start waiting to join one
            emit("waiting", {"in_game": False})
        else:
            # Game was found so join it
            with game.lock:
                join_room(game.id)
                set_curr_room(user_id, game.id)
                game.add_player(user_id, username_id=username_id, user_type=user_type)
                # game.add_player(user_id)

                if game.is_ready():
                    # Game is ready to begin play
                    game.activate()
                    ACTIVE_GAMES.add(game.id)
                    emit(
                        "start_game",
                        {"spectating": False, "start_info": game.to_json()},
                        room=game.id,
                    )
                    socketio.start_background_task(play_game, game)
                else:
                    # Still need to keep waiting for players
                    WAITING_GAMES.put(game.id)
                    emit("waiting", {"in_game": True}, room=game.id)


@socketio.on("leave")
def on_leave(data):
    user_id = request.sid
    with USERS[user_id]:
        was_active = _leave_game(user_id)

        if was_active:
            emit("end_game", {"status": Game.Status.DONE, "data": {}})
        else:
            emit("end_lobby")


class UserActionTracker:
    def __init__(self):
        self.user_last_action_time = {}

    def should_process_action(self, user_id, debounce_period=(1/60)):
        """Check if the action for the given user_id should be processed."""
        current_time = time.time()
        last_action_time = self.user_last_action_time.get(user_id, 0)

        if current_time - last_action_time > debounce_period:
            self.user_last_action_time[user_id] = current_time
            return True
        return False

user_action_tracker = UserActionTracker()

@socketio.on("action")
def on_action(data):
    user_id = request.sid
    action = data["action"]

    if not user_action_tracker.should_process_action(user_id):
        return

    game = get_curr_game(user_id)
    if not game:
        return
    
    game.enqueue_action(user_id, action)


@socketio.on("connect")
def on_connect():
    print("connect triggered", file=sys.stderr)    
    user_id = request.sid
    username_id = get_user_id()    
    username_type = get_user_type()    

    if username_id and username_id in ACTIVE_SESSIONS.values():
        reason = 'multiple_browsers'
        socketio.emit('error_page', {'reason': reason}, room=user_id)
        socketio.emit('initiate_disconnect', room=user_id)
        return

    if len(USERS) >= max_users:
        reason = 'server_full'
        socketio.emit('error_page', {'reason': reason}, room=user_id)
        socketio.emit('initiate_disconnect', room=user_id)
        return
    
    if user_id in USERS:
        return

    USERS[user_id] = Lock()
    ACTIVE_SESSIONS[user_id] = username_id
    ACTIVE_SESSIONS_UserTypes[user_id] = username_type
    
    if username_type is None or username_id is None: 
        username_type_id = None
    else:
        username_type_id = username_type + username_id
    
    if username_type_id in USERNAME_TIMERS:
        USERNAME_TIMERS[username_type_id].cancel()
        del USERNAME_TIMERS[username_type_id]    


@socketio.on("disconnect")
def on_disconnect():
    print("disconnect triggered", file=sys.stderr)
    
    user_id = request.sid
    username_id = get_user_id()
    username_type = get_user_type()
    log_event(username_id, username_type,"disconnect","disconnect Triggered")       
    
    if user_id not in USERS:
        return
    
    with USERS[user_id]:
        _leave_game(user_id)

    del USERS[user_id]
    del ACTIVE_SESSIONS[user_id]
    del ACTIVE_SESSIONS_UserTypes[user_id]
    
    if username_type is None or username_id is None: 
        username_type_id = None
    else:
        username_type_id = username_type + username_id
    
    USERNAME_TIMERS[username_type_id] = Timer(USERNAME_TIMEOUT_SECONDS, remove_user_logs, [username_type_id])
    USERNAME_TIMERS[username_type_id].start()    


def remove_user_logs(username_type_id):
    """
    Function to remove user logs.
    """
    if username_type_id in LOG_USERS:

        del LOG_USERS[username_type_id]
        del USER_SCORES[username_type_id]
        


# Exit handler for server
def on_exit():
    # Force-terminate all games on server termination
    for game_id in GAMES:
        socketio.emit(
            "end_game",
            {
                "status": Game.Status.INACTIVE,
            },
            room=game_id,
        )


#############
# Game Loop #
#############

    
def play_game(game: OvercookedGame, fps=6):
    """
    Asynchronously apply real-time game updates and broadcast state to all clients currently active
    in the game. Note that this loop must be initiated by a parallel thread for each active game

    game (Game object):     Stores relevant game state. Note that the game id is the same as to socketio
                            room id for all clients connected to this game
    fps (int):              Number of game ticks that should happen every second
    """
    import time
    fps = 10
    frame_duration = 1.0 / fps
    status = Game.Status.ACTIVE
    while status != Game.Status.DONE and status != Game.Status.INACTIVE:
        frame_start_time = time.time()
        with game.lock:
            start = time.time()
            status = game.tick()
        if status == Game.Status.RESET:

            socketio.emit(
                "reset_game",
                {
                    "state": game.to_json(),
                    "timeout": game.reset_timeout,
                },
                room=game.id,
            )
            socketio.sleep(game.reset_timeout / 1000)
        else:
            socketio.emit(
                "state_pong", {"state": game.get_state()}, room=game.id
            )
        frame_end_time = time.time()
        elapsed_time = frame_end_time - frame_start_time
        time_to_sleep = max(0, frame_duration - elapsed_time)
        socketio.sleep(time_to_sleep)

    with game.lock:

        if status != Game.Status.INACTIVE:
            game.deactivate()
        cleanup_game(game)


if __name__ == "__main__":
    # Dynamically parse host and port from environment variables (set by docker build)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 80))

    # Attach exit handler to ensure graceful shutdown
    atexit.register(on_exit)

    # https://localhost:80 is external facing address regardless of build environment
    socketio.run(app, host=host, port=port, log_output=app.config["DEBUG"])
