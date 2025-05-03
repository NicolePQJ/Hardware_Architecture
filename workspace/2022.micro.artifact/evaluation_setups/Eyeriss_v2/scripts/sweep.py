import os, inspect, sys, subprocess, yaml, pprint, math, pickle, shutil, signal, math, time, argparse
from copy import deepcopy
#import pytimeloop.timeloopfe.v4 as tl

# static paths
this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
this_directory = os.path.dirname(this_file_path)
os.chdir(this_directory)

# paths to important input specs
arch_file_path = os.path.join(this_directory, "..", "architecture", "new_arch.yaml")
components_file_path = os.path.join(this_directory, "..", "architecture", "components.yaml")
sparse_iact_opt_file_path = os.path.join(this_directory, "..", "sparse_opt", "sparse_iact_opt.yaml")
dense_iact_opt_file_path = os.path.join(this_directory, "..", "sparse_opt", "dense_iact_opt.yaml")
workload_dir_path = os.path.join(this_directory, "..", "workload_alexnet")
ert_path = os.path.join(this_directory, "..", "ert_art", "ERT.yaml")
art_path = os.path.join(this_directory, "..", "ert_art", "ART.yaml")
mappings_dir = os.path.join(this_directory, "..", "mappings_found")
dataflow_file_path = os.path.join(this_directory, "..", "dataflow", "row_stationary.yaml")
mapper_file_path = os.path.join(this_directory, "..", "mapper", "mapper.yaml")

def capture_log(cmd, log_path, timeout=600):
    """
    Run *cmd* while streaming **both** stdout and stderr into *log_path*
    and the console.  Return the subprocess' exit code.
    """
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                bufsize=1,
                                universal_newlines=True)

        start = time.time()
        for line in proc.stdout:
            print(line, end="")    # mirror to console
            lf.write(line)

        try:
            proc.wait(timeout=max(0, timeout - (time.time() - start)))
        except subprocess.TimeoutExpired:
            proc.kill()
            lf.write(f"\n[run timed‑out after {timeout}s]\n")
            return -1
    return proc.returncode

def run_timeloop(job_name, input_dict, ert_path, art_path, base_dir):
    
    print("Running job: ", job_name)
    output_dir = os.path.join(base_dir, "output")
    
    if not os.path.exists(output_dir) \
       and not os.path.exists(os.path.join(output_dir, "timeloop-mapper.map+stats.xml")) \
       and not os.path.exists(os.path.join(output_dir, "timeloop-model.map+stats.xml")):
       os.makedirs(output_dir)
    else:
        if not OVERWRITE:
            print("Found existing results: ", output_dir)
            return True
        else:
            print("Found and overwrite existing results: ", output_dir)
    
    input_file_path = os.path.join(base_dir, "aggregated_input.yaml")
    shutil.copy(ert_path, os.path.join(base_dir, "ERT.yaml"))
    shutil.copy(art_path, os.path.join(base_dir, "ART.yaml"))
    
    if not USE_MODEL: 
        input_dict.pop("mapping", 0)
        yaml.dump(input_dict, open(input_file_path, "w"), default_flow_style=False)
        os.chdir(output_dir)
        #subprocess_cmd = ["timeloop-mapper", input_file_path, os.path.join(base_dir, "ERT.yaml"), os.path.join(base_dir, "ART.yaml")]
        subprocess_cmd = ["timeloop-mapper", input_file_path]

        print("\tRunning test: ", job_name)

        p = subprocess.Popen(subprocess_cmd)
        try:
            p.communicate(timeout=300) # wait for at most 5 min
#             p.communicate(timeout=1200) # wait for at most 20 min
        except KeyboardInterrupt:
           p = 0
           while p <= 60 and not os.path.exists(os.path.join(output_dir, "timeloop-mapper.map+stats.xml")):
               time.sleep(1)
               p = p + 1        
        except subprocess.TimeoutExpired:
           print(job_name,  " reaches timeout limit")
           os.kill(p.pid, signal.SIGINT)
        
        return True
    
    else:
        model_input_dict = deepcopy(input_dict)
        model_input_dict.pop("architecture_constraints")
        model_input_dict.pop("mapspace_constraints")
        model_input_dict.pop("mapper")
        yaml.dump(model_input_dict, open(input_file_path, "w"), default_flow_style=False)
        
        os.chdir(output_dir)
        #subprocess_cmd = ["timeloop-model", input_file_path, os.path.join(base_dir, "ERT.yaml"), os.path.join(base_dir, "ART.yaml"), os.path.join(base_dir, "map.yaml")]
        subprocess_cmd = ["timeloop-model", input_file_path, os.path.join(base_dir, "map.yaml")]
        p = subprocess.Popen(subprocess_cmd)

# def run_timeloop(job_name, input_dict, base_dir, use_model=False, overwrite=False):
#     """
#     Build <base_dir>/aggregated_input.yaml and run timeloop‑mapper (or
#     timeloop‑model) once.  All console output goes into <job_name>.log.
#     """
#     os.environ.setdefault(
#         "ACCELERGY_COMPONENT_LIBRARIES",
#         os.path.normpath(os.path.join(this_directory, "..", "architecture"))
#     )
#     print(f"=== {job_name} ===")

#     output_dir = os.path.join(base_dir, "output")
#     os.makedirs(output_dir, exist_ok=True)

#     stats_done = any(fname.endswith(".map+stats.xml")
#                      for fname in os.listdir(output_dir))
#     if stats_done and not overwrite:
#         print("  ↪ results already exist — skipping")
#         return True

#     # ------------------------------------------------------------------
#     # 1.  dump the amalgamated YAML
#     # ------------------------------------------------------------------
#     input_file = os.path.join(base_dir, "aggregated_input.yaml")
#     with open(input_file, "w") as fh:
#         yaml.dump(input_dict, fh, default_flow_style=False)

#     # ------------------------------------------------------------------
#     # 2.  build the command line
#     # ------------------------------------------------------------------
#     if use_model:
#         exe = "timeloop-model"
#     else:
#         exe = "timeloop-mapper"
#     #os.chdir(output_dir)
#     #cmd = [exe, input_file, "-v", "1", "--no-progress-bar"]
#     cmd = [exe,
#         input_file,
#         "-o", output_dir,          # tell Timeloop where to write files
#         "-v", "1", "--no-progress-bar"]

#     log_file = os.path.join(base_dir, f"{job_name}.log")
#     print(f"  ↪ running {exe}, log → {log_file}")
#     rc = capture_log(cmd, log_file, timeout=600)          ### NEW

#     if rc != 0:
#         print(f"  ✗ {exe} exited with code {rc} — see log")
#         return False

#     # quick sanity: did we get a stats file?
#     stats_ok = any(fname.endswith(".map+stats.xml")
#                    for fname in os.listdir(output_dir))
#     if not stats_ok:
#         print("  ✗ no stats file produced — mapper found no valid mapping")
#         return False

#     print("  ✓ done")
#     return True

def no_op_constructor(loader, tag_suffix, node):
    """
    A catch-all constructor that ignores the custom tag
    and just constructs the node as a normal Python dict
    (or list, etc.) using safe_load defaults.
    """
    return loader.construct_object(node, deep=True)

def main():
    yaml.SafeLoader.add_multi_constructor('', no_op_constructor)
    arch_spec = yaml.load(open(arch_file_path), Loader = yaml.SafeLoader)
    component_spec = yaml.load(open(components_file_path), Loader = yaml.SafeLoader)
    constraints_spec = yaml.load(open(dataflow_file_path), Loader = yaml.SafeLoader)
    mapper_spec = yaml.load(open(mapper_file_path), Loader = yaml.SafeLoader)

    stats_collector = {}

    for layer in os.listdir(workload_dir_path):
        aggregated_input = {}
        
        full_path = os.path.join(workload_dir_path, layer)
        if os.path.isfile(full_path) and layer.endswith(('.yaml', '.yml')):
            workload_spec = yaml.load(open(full_path), Loader=yaml.SafeLoader)
        else:
            continue
#         workload_spec = yaml.load(open(os.path.join(workload_dir_path, layer)), Loader = yaml.SafeLoader)
        print(full_path)
        print(workload_spec["problem"]["instance"]["densities"]["Inputs"])
        density = workload_spec["problem"]["instance"]["densities"]["Inputs"]
        print(f"{layer=}, {density=}")
        dense_iact = density > 0.9

        if not dense_iact:             
            sparse_opt_spec = yaml.load (open(sparse_iact_opt_file_path), Loader = yaml.SafeLoader)
        else:
            sparse_opt_spec = yaml.load (open(dense_iact_opt_file_path), Loader = yaml.SafeLoader)
       
        mapping_file_path = os.path.join(mappings_dir, layer)
#         mapping_spec = yaml.load(open(mapping_file_path), Loader = yaml.SafeLoader)
        
        aggregated_input.update(arch_spec)
        aggregated_input.update(component_spec)
        aggregated_input.update(sparse_opt_spec)
        aggregated_input.update(workload_spec)
#         aggregated_input.update(mapping_spec)
        aggregated_input.update(constraints_spec)
        aggregated_input.update(mapper_spec)

        job_name = layer.split('.')[0]
        base_output_dir = os.path.join(OUT_DIR, job_name)

        # run evaluation 
        run_timeloop(job_name, aggregated_input, ert_path, art_path, base_output_dir)
        # ok = run_timeloop(job_name,
        #                   aggregated_input,
        #                   base_output_dir,
        #                   use_model = USE_MODEL,
        #                   overwrite  = OVERWRITE)
        # if not ok:
        #     print(f"Stopping sweep after failure in layer {job_name}")
        #     break

    
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser("sweep alexnet conv layers to get DRAM compression ratio for Eyeriss. Usage: python3 run_alexnet_conv.py")
    parser.add_argument('-o', '--output_dir', type=str, default=os.path.join(this_directory, "..", "outputs", "density_0.01_02"), help='abs path to top level output directory' )
    parser.add_argument('--max_layers', type=int, default=100, help='max number of layers to run')
    parser.add_argument('--no_overwrite', action="store_true", help='skip job there is already some previous results in the output folder')
    parser.add_argument('--search_mapping', action="store_true", help='search for optimal mapping instead of using the provided mappings, this option will make the experiment run much slower')
    parser.add_argument('--workload_path', type=str, default="workload_alexnet", help='use a workload other than the default alexNet')
    options = parser.parse_args()
   
    OUT_DIR = options.output_dir
    OVERWRITE = not options.no_overwrite 
    USE_MODEL = not options.search_mapping 
    workload_dir_path = os.path.join(this_directory, "..", options.workload_path)
    
    main()

