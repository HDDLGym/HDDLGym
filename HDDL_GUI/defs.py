PROBLEMS = {
    "zeno5":        ("NumericTCORE/benchmark/ZenoTravel-no-constraint/domain.pddl",         "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile5.pddl"),
    "zeno5_bis":    ("NumericTCORE/benchmark/ZenoTravel-no-constraint/domain.pddl",         "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile5_bis.pddl"),
    "zeno5_bis_n":  ("NumericTCORE/benchmark/ZenoTravel-n/domain_with_n.pddl",              "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile5_bis.pddl"),
    "zeno8":        ("NumericTCORE/benchmark/ZenoTravel-no-constraint/domain.pddl",         "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile8.pddl"),
    "zeno12":       ("NumericTCORE/benchmark/ZenoTravel-no-constraint/domain.pddl",         "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile12.pddl"),
    "zeno12_n":     ("NumericTCORE/benchmark/ZenoTravel-n/domain_with_n.pddl",              "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile12.pddl"),
    "zeno15":       ("NumericTCORE/benchmark/ZenoTravel-no-constraint/domain.pddl",         "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile15.pddl"),
    "zeno23":       ("NumericTCORE/benchmark/ZenoTravel-no-constraint/domain.pddl",         "NumericTCORE/benchmark/ZenoTravel-no-constraint/pfile23.pddl"),
    "sailing1":     ("ENHSP-Public/ijcai18_benchmarks/sailing_ln/domain.pddl",              "ENHSP-Public/ijcai18_benchmarks/sailing_ln/instance_1_1_1229.pddl"),
    "satellite1":   ("ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/metricSat.pddl",    "ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/pfile1"),
    "satellite3":   ("ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/metricSat.pddl",    "ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/pfile3"),
    "satellite4":   ("ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/metricSat.pddl",    "ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/pfile4"),
    "satellite5":   ("ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/metricSat.pddl",    "ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/pfile5"),
    "satellite15":  ("ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/metricSat.pddl",    "ENHSP-Public/ijcai16_benchmarks/Satellite/Numeric/pfile15"),
    "rover1":       ("ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/NumRover.pddl",         "ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/pfile1"),
    "rover3":       ("ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/NumRover.pddl",         "ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/pfile3"),
    "rover3_bis":   ("ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/NumRover.pddl",         "ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/pfile3_bis"),
    "rover4":       ("ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/NumRover.pddl",         "ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/pfile4"),
    "rover10":      ("ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/NumRover.pddl",         "ENHSP-Public/ijcai16_benchmarks/Rover-Numeric/pfile10"),
    "sar1":         ("ENHSP-Public/sar/handcraft/domain.pddl",                              "ENHSP-Public/handcraft/pfile1.pddl"),
    # HDDL:
    "zeno5_hddl":   ("HDDL_env/zeno_domain.hddl",       "HDDL_env/zeno_problem_5.hddl"),
    "monroe_hddl": ("HDDL_env/monroe_hddl.hddl",     "HDDL_env/monroe_problem.hddl"),
}

KNOWN_PROBLEMS_STR = 'PROBLEM_NAME: [' + ', '.join(PROBLEMS) + ']'

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

COMPILED_DOMAIN_PATH = "tmp/compiled_dom.pddl"
COMPILED_PROBLEM_PATH = "tmp/compiled_prob.pddl"
UPDATED_PROBLEM_PATH = "tmp/updatedProblem.pddl"
UPDATED_HDDL_DOMAIN_PATH = "tmp/updatedDomain.hddl"

class PlanFiles:
    ORIGINAL = 'original'
    COMPILED = 'compiled'
    PATH = 'path'

class PlanMode:
    OPTIMAL = 'opt-hrmax'
    SATISFICING = 'sat-hmrp'
    ANYTIME = 'anytime'
    ANYTIMEAUTO = 'anytimeAuto'
    DEFAULT = 'def'

class NtcoreStrategy:
    NAIVE = 'naive'
    REGRESSION = 'regression'
    DELTA = 'delta'
    
    
SHELL_PRINTS = False
GUI_PROMPT = True

prompt = lambda x: None

def mprint(txt):
    if SHELL_PRINTS:
        print(txt)
    if GUI_PROMPT:
        txt = txt.replace(color.BOLD, '')
        txt = txt.replace(color.END, '')
        prompt(txt)
def setPromptFunction(prompt_function):
    global prompt
    prompt = prompt_function