## HDDL Domain and Problem Files
This folder contains domains that can be used for HDDLGym, including existing domains from IPC (under ipc2023_domains folder), and Overcooked domain

It also includes a supportive python code to help modify and/or verify the HDDL domain so it can work with HDDLGym. To run this code, use the following command:

``` python modify_hddl_domain_for_HDDLGym.py --domain <path/to/domain/file> --new-domain <path/to/new/domain/file> ```

- <path/to/domain/file> is the directory to the HDDL domain file that you want to modify or verify.

- <path/to/new/domain/file> is the directory to save the modified or verified domain file.

For example:

``` python modify_hddl_domain_for_HDDLGym.py --domain ./test_input_modify_overcooked.hddl --new-domain ./test_output_modify_overcooked.hddl ```
