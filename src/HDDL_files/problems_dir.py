# This file combines domain and problem directories

PROBLEMS = {
    # Overcooked domain:
    "overcooked_2agents_collab": ("HDDL_files/Custom_environments/Overcooked_specialization/overcooked_short_domain.hddl",\
                                   "HDDL_files/Custom_environments/Overcooked_specialization/overcooked_short_prob2.hddl"),
    "overcooked_3agents_collab": ("HDDL_files/Custom_environments/Overcooked_specialization/overcooked_short_domain.hddl",\
                                   "HDDL_files/Custom_environments/Overcooked_specialization/overcooked_short_prob3.hddl"),

    # Transport domain:
    "transport_1agent_collab": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab.hddl",\
                                  "HDDL_files/ipc2023_domains/Transport/pfile01.hddl"),
    "transport_1agent_collab_folder": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab.hddl",\
                                  "HDDL_files/ipc2023_domains/Transport/problem_folder"),
    "transport_1agent_no_collab": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_without_collab.hddl",\
                                  "HDDL_files/ipc2023_domains/Transport/pfile01.hddl"),
    "transport_2agents_collab": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab.hddl",\
                                  "HDDL_files/ipc2023_domains/Transport/pfile11.hddl"),
    "transport_2agents_no_collab": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_without_collab.hddl",\
                                  "HDDL_files/ipc2023_domains/Transport/pfile11.hddl"),
    "transport_2agents_heterogeneous": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab.hddl",\
                                        "HDDL_files/ipc2023_domains/Transport/transport_hetero_2agents_pfile02.hddl"),
    
    "transport_2agents_heterogeneous_without_collabmethod": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab_no_collabmethod.hddl",\
                                        "HDDL_files/ipc2023_domains/Transport/transport_hetero_2agents_pfile02.hddl"),
    "transport_3agents_collab": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab.hddl",\
                                  "HDDL_files/ipc2023_domains/Transport/pfile21.hddl"),
    "transport_3agents_no_collab": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_without_collab.hddl",\
                                  "HDDL_files/ipc2023_domains/Transport/pfile21.hddl"),
    "transport_2agents_heterogeneous_with_noop": ("HDDL_files/ipc2023_domains/Transport/transport_domain_hddlgym_with_collab_with_noop.hddl",\
                                        "HDDL_files/ipc2023_domains/Transport/transport_collab_pfile01.hddl"),
    
    # Rover domain:
    "rover_1agent": ("HDDL_files/ipc2023_domains/Rover/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Rover/pfile01.hddl"),
    "rover_2agents": ("HDDL_files/ipc2023_domains/Rover/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Rover/pfile03.hddl"),
    "rover_3agents": ("HDDL_files/ipc2023_domains/Rover/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Rover/pfile07.hddl"),
    "rover_4agents": ("HDDL_files/ipc2023_domains/Rover/domain_hddlgym.hddl",\
                            "HDDL_files/ipc2023_domains/Rover/pfile09.hddl"),

    # Satellite domain:
    "satellite_1obs_1sat_1mod": ("HDDL_files/ipc2023_domains/Satellite/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Satellite/1obs-1sat-1mod.hddl"),
    "satellite_2obs_2sat_1mod": ("HDDL_files/ipc2023_domains/Satellite/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Satellite/2obs-2sat-1mod.hddl"),
    "satellite_3obs_3sat_1mod": ("HDDL_files/ipc2023_domains/Satellite/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Satellite/3obs-3sat-1mod.hddl"),
    "satellite_4obs_4sat_4mod": ("HDDL_files/ipc2023_domains/Satellite/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Satellite/4obs-4sat-4mod.hddl"),

    # Barman-BDI domain:
    "barman_bdi_2agents_1drink": ("HDDL_files/ipc2023_domains/Barman-BDI/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Barman-BDI/pfile01.hddl"),
    "barman_bdi_2agents_2drinks": ("HDDL_files/ipc2023_domains/Barman-BDI/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Barman-BDI/pfile02.hddl"),
    "barman_bdi_2agents_3drinks": ("HDDL_files/ipc2023_domains/Barman-BDI/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Barman-BDI/pfile03.hddl"),

    # Depots domain:
    "depots_1agent": ("HDDL_files/ipc2023_domains/Depots/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Depots/pfile01.hddl"),
    
    # Zenotravel domain:
    "zenotravel_1agent": ("HDDL_files/ipc2023_domains/Zenotravel/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Zenotravel/pfile01.hddl"),
    "zenotravel_2agents": ("HDDL_files/ipc2023_domains/Zenotravel/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Zenotravel/pfile03.hddl"),
    
    # Factories-simple domain:
    "factories_simple_1agent": ("HDDL_files/ipc2023_domains/Factories-simple/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Factories-simple/pfile01.hddl"),
    
    # Minecraft-Player domain:
    "minecraft_player_1agent_003": ("HDDL_files/ipc2023_domains/Minecraft-Player/domain_hddlgym.hddl",\
                          "HDDL_files/ipc2023_domains/Minecraft-Player/p-003-003-003-003.hddl"),

    # Search-and-Rescue domain:
    "search_and_rescue_1agent": ("HDDL_files/Custom_environments/SearchAndRescue/SAR_domain.hddl",\
                            "HDDL_files/Custom_environments/SearchAndRescue/SAR_problem.hddl"),
    
    # Taxi domain:
    "taxi_2agents": ("HDDL_files/Custom_environments/Taxi/domain.hddl",\
                            "HDDL_files/Custom_environments/Taxi/problem1.hddl"),

}