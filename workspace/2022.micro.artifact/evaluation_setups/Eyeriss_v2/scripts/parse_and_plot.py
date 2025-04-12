import os, inspect, sys, subprocess, yaml, pprint, math, pickle, shutil, signal, math, time, argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

# static paths
this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
os.chdir(this_directory)
workload = 0

# import timeloop result parser
sys.path.append(os.path.join(this_directory, "..", "..", "utils"))
from parse_timeloop_output import parse_timeloop_stats
# from sweep import workload_dir_path

# static paths
this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
os.chdir(this_directory)

# Extract memory accesses from txt file (cannot find this information in parsed file)
def memory_accesses(file_path):
    components = {"psum_spad", "weight_spad", "iact_spad", "glb_psum", "glb_iact", "DRAM"}
    results = {}
    curr = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line[:3] == "===" and line [-3:] == "===":
                component = line.strip("= ")
                if component in components:
                    curr = component
            if curr in components and "Total scalar accesses" in line:
                parts = line.split(":")
                results[curr] = int(parts[1].strip())

    return results

def main(stats_prefix):
    stats_collector = {}
    
    if workload == "workload_alexnet":
        job_names = ["alexnet_conv" + str(i) for i in range(1, 6)]
    else:
        job_names = ["mobilenet_conv" + str(i) for i in range(1, 9)]
    
    # collect all the static job information
    for layer in os.listdir(workload_dir_path):
        job_name = layer.split('.')[0]
        base_output_dir = os.path.join(OUT_DIR, job_name)
     
        full_path = os.path.join(workload_dir_path, layer)
        if os.path.isfile(full_path) and layer.endswith(('.yaml', '.yml')):
            workload_spec = yaml.load(open(full_path), Loader=yaml.SafeLoader)
        else:
            continue
#         workload_spec = yaml.load(open(os.path.join(workload_dir_path, layer)), Loader = yaml.SafeLoader)
        dense_iact = workload_spec["problem"]["instance"]["densities"]["Inputs"] > 0.9
        stats_collector[job_name] = {"path": base_output_dir, "dense_iact": dense_iact}
    
    # find output files and parse based on the collected information
    energy_values = {}
    memory_values = {}
    print("Eyeriss DRAM Compression Ratioes (read + write)")
    for job_name in job_names:
        job_info = stats_collector[job_name]

        memory_access_stats = memory_accesses(os.path.join(job_info["path"], "output", stats_prefix + ".stats.txt"))
        memory_values[job_name] = memory_access_stats
        
#         baseline_dense_path = job_info["path"].replace("outputs", "dense_outputs")
#         baseline_dense_output_stats = parse_timeloop_stats(os.path.join(baseline_dense_path, "output", stats_prefix + ".map+stats.xml"))
#         baseline_DRAM_accesses = sum(j for j in baseline_dense_output_stats["energy_breakdown_pJ"]["DRAM"]["actual_accesses_per_instance"])
        
        job_output_stats = parse_timeloop_stats(os.path.join(job_info["path"], "output", stats_prefix + ".map+stats.xml"))
        output_path = os.path.join(job_info["path"], "output", stats_prefix + "parsed_output.txt")
        with open(output_path, 'w') as f:
            f.write(str(job_output_stats))
        energy = {}

        for component, values in job_output_stats['energy_breakdown_pJ'].items():
            energy[component] = values['energy']
        
        energy_values[job_name] = energy
        job_DRAM_weight_accesses =  job_output_stats["energy_breakdown_pJ"]["DRAM"]["actual_accesses_per_instance"][0]
        job_DRAM_iact_accesses =  job_output_stats["energy_breakdown_pJ"]["DRAM"]["actual_accesses_per_instance"][1]
        job_DRAM_oact_accesses =  job_output_stats["energy_breakdown_pJ"]["DRAM"]["actual_accesses_per_instance"][2]
       
        # ~0.76 captures the overhead and is obtained from the stats files as metadata overhead ratio
        # Please look into the timeloop.stats.txt files in the outputs folder for information
        if job_info["dense_iact"]: 
            eyeriss_DRAM_accesses = job_DRAM_weight_accesses + job_DRAM_iact_accesses + job_DRAM_oact_accesses/0.76
        else:
            eyeriss_DRAM_accesses = job_DRAM_weight_accesses + job_DRAM_iact_accesses/0.76 + job_DRAM_oact_accesses/0.76
        
#         print(job_name, ": ",  round(baseline_DRAM_accesses/eyeriss_DRAM_accesses, 1))
#         print(job_output_stats)
#         for k in job_output_stats.keys():
#             print(k, job_output_stats[k])
#             print('-----------------------------------------------')
#         print('------------------------------------------------------------------------------------------------------')
    normalized_energy_data_per_job = {}
    energy_data_per_job = {}
    total_energy_usage = {}
    for job, energy_data in energy_values.items():
        total_energy = sum(energy_data.values())
        total_energy_usage[job] = total_energy
        normalized_energy_data_per_job[job] = {key: value / total_energy for key, value in energy_data.items()}
        energy_data_per_job[job] = {key: value for key, value in energy_data.items()}

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define component names (same for all jobs)
    components = list(energy_values[job_names[0]].keys())

    # Plot the stacked bars for each job
    bottoms = {job: 0 for job in job_names}  # Start from 0 for each job
    for i, component in enumerate(components):
        # Get the energy values for the current component across all jobs
        energy_value = [normalized_energy_data_per_job[job][component] for job in job_names]

        # Plot the bars for this component
        for j, job in enumerate(job_names):
            ax.bar(job, energy_value[j], bottom=bottoms[job], color=plt.cm.tab10(i), label=component if j == 0 else "")
            bottoms[job] += energy_value[j]  # Update the bottom for the next component

    # Set labels and title
    ax.set_ylabel('Normalized Energy (0.0 to 1.0)')
    ax.set_xlabel('Job Name')
    ax.set_title('Energy Breakdown for Different Jobs')
    ax.set_ylim(0, 1)

    # Add a legend
    ax.legend(title="Components", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
#     plt.tight_layout()
#     plt.show()
    output_fig_path = os.path.join(job_info["path"], "output", "energy_breakdown_plot.png")
    plt.savefig(output_fig_path, bbox_inches='tight')
    plt.close()
    print("Saved:", output_fig_path)
    
    #plot energy usage
    jobs = list(total_energy_usage.keys())
    energy = list(total_energy_usage.values())
    fig, ax = plt.subplots(figsize=(12, 6))  # You can adjust the figure size as needed
#     ax.bar(jobs, energy, color='skyblue')
    
    
    bottoms = {job: 0 for job in job_names}  # Start from 0 for each job
    for i, component in enumerate(components):
        # Get the energy values for the current component across all jobs
        energy_value = [energy_data_per_job[job][component] for job in job_names]

        # Plot the bars for this component
        for j, job in enumerate(job_names):
            ax.bar(job, energy_value[j], bottom=bottoms[job], color=plt.cm.tab10(i), label=component if j == 0 else "")
            bottoms[job] += energy_value[j]  # Update the bottom for the next component

    # Add labels and title
    ax.set_xlabel('Job Name')
    ax.set_ylabel('Total Energy Usage')
    ax.set_title('Total Energy Usage per Job')
    ax.legend(title="Components", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_fig_2_path = os.path.join(job_info["path"], "output", "total_energy_plot.png")
    plt.savefig(output_fig_2_path, bbox_inches='tight')
    plt.close()
    print("Saved:", output_fig_2_path)
    
    #plot memory accesses
    jobs = list(memory_values.keys())
    memory = list(memory_values.values())
    components = list(memory_values[jobs[0]].keys())
    fig, ax = plt.subplots(figsize=(12, 6))  # You can adjust the figure size as needed
    
    bottoms = {job: 0 for job in job_names}  # Start from 0 for each job
    for i, component in enumerate(components):
        # Get the energy values for the current component across all jobs
        memory_access = [memory_values[job][component] for job in job_names]

        # Plot the bars for this component
        for j, job in enumerate(job_names):
            ax.bar(job, memory_access[j], bottom=bottoms[job], color=plt.cm.tab10(i), label=component if j == 0 else "")
            bottoms[job] += memory_access[j]  # Update the bottom for the next component

    # Add labels and title
    ax.set_xlabel('Job Name')
    ax.set_ylabel('Total Memory Accesses')
    ax.set_title('Total Memory Accesses per Job')
    ax.legend(title="Components", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    print(job_info["path"])
    output_fig_3_path = os.path.join(job_info["path"], "output", "total_memory_plot.png")
    plt.savefig(output_fig_3_path, bbox_inches='tight')
    plt.close()
    print("Saved:", output_fig_3_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("parse result to get DRAM compression ratio for Eyeriss. Usage: python3 parse_and_plot.py")
    parser.add_argument('--stats_prefix', type=str, default="timeloop-model", help='the output prefix that the parser to be looking for' )
    parser.add_argument('-o', '--output_dir', type=str, default=os.path.join(this_directory, "..", "outputs"), help='abs path to top level output directory that needs to be parsed' )
    parser.add_argument('--workload_path', type=str, default="workload_alexnet", help='use a workload other than the default alexNet')
    options = parser.parse_args()
    OUT_DIR = options.output_dir
    workload = options.workload_path
    workload_dir_path = os.path.join(this_directory, "..", options.workload_path)
    
    main(options.stats_prefix)
