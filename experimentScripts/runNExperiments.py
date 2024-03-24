import os
import shutil
from multiprocessing import Pool
import matplotlib.pyplot as plt
import benchmarksUtil
import inputOptions
import argparse
from ParallelRange import get_thread_allocations


alphaValues = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# alphaValues = [1.0, 0.5, 0.0]
iterSamples = [1000, 10000]

def get_exec_command_from_input():
    global input_values
    dir_path = os.path.dirname(os.path.realpath(__file__))
    command = 'python3 ' + dir_path + '/runExperiments.py'
    if input_values.information_level:
        command += ' ' + inputOptions.information_level_option + ' ' + input_values.information_level
    if input_values.update_method:
        command += ' ' + inputOptions.update_method_option + ' ' + input_values.update_method
    if input_values.simulate_mec:
        command += ' ' + inputOptions.simulate_mec_option + ' ' + input_values.simulate_mec
    if input_values.get_error_probability:
        command += ' ' + inputOptions.get_error_probability_option
    if input_values.delta_t_method:
        command += ' ' + inputOptions.deltat_method_option + ' ' + input_values.delta_t_method
    if input_values.is_ctmdp:
        command += ' ' + inputOptions.ctmdp_benchmarks_option
    return command


def run_benchmark_iteration(i):
    global input_values
    global exec_command
    global number_of_threads_in_exec

    output_directory_option = inputOptions.output_directory_option + ' ' + input_values.output_directory + '/' + f'iteration{i}'
    n_threads_option = inputOptions.number_of_threads_option + ' ' + str(number_of_threads_in_exec)
    os.system(exec_command + ' ' + output_directory_option + ' ' + n_threads_option)


def run_benchmarks(n):
    for i in range(n):
        run_benchmark_iteration(i)


# For running iterations in parallel
def run_benchmarks_range(parallel_range):
    for i in range(parallel_range.start, parallel_range.end + 1):
        run_benchmark_iteration(i)


def run_benchmarks_in_parallel(n):
    global input_values
    parallel_ranges = get_thread_allocations(input_values.number_of_threads, n)

    pool = Pool(processes=input_values.number_of_threads)
    pool.map(run_benchmarks_range, parallel_ranges)
    pool.close()
    pool.join()


def schedule_and_run_benchmarks(n):
    global input_values
    global number_of_threads_in_exec

    if input_values.number_of_threads == 1 or n == 1:
        run_benchmarks(n)
        return

    number_of_threads_in_exec = 1
    run_benchmarks_in_parallel(n)


def result_comparator(model_result):
    return model_result.get_bounds_diff()


def write_model_result(result_file, model_result):
    result_file.write('Execution time: ' + str(model_result.get_runtime()) + '\n')
    result_file.write('Lower bound: ' + str(model_result.lower_bounds[-1]) + '\n')
    result_file.write('Upper bound: ' + str(model_result.upper_bounds[-1]) + '\n')
    result_file.write('Iteration number: ' + str(model_result.iteration_number) + '\n')
    result_file.write('Num states explored: ' + str(model_result.num_explored_states) + '\n')
    result_file.write('Total number of samples: ' + str(model_result.total_samples) + '\n')
    result_file.write('Total number of simulations: ' + str(model_result.total_simulations) + '\n')
    result_file.write('Value of iterSamples: ' + str(model_result.iter_sample) + '\n')
    result_file.write('Value of alpha: ' + str(model_result.alpha) + '\n')
    result_file.write('\n')
    result_file.write('\n')
    result_file.write('\n')


def write_model_results(model_name, model_result_list, alpha_value, result_directory):
    print(model_name)
    result_file_name = f"{result_directory}{model_name}-{alpha_value}.txt"


    with open(result_file_name, 'w') as result_file:
        for model_result in model_result_list:
            write_model_result(result_file, model_result)

        for iterSample in iterSamples:

            result_file.write('IterSample: ' + str(iterSample) + '\n')
            filteredOutput = [y for y in model_result_list if y.iter_sample == iterSample]

            (al, au, ar, av_states, av_total_samples, av_total_simulations) = benchmarksUtil.get_average_values(filteredOutput)
            precision = au - al
            result_file.write('Average Lower Bound: ' + str(al) + '\n')
            result_file.write('Average Upper Bound: ' + str(au) + '\n')
            result_file.write('Average Run time: ' + str(ar) + '\n')
            result_file.write('Precision: ' + str(precision) + '\n')
            result_file.write('Average number of states explored: ' + str(av_states) + '\n')
            result_file.write('Average number of samples per iteration: ' + str(av_total_samples) + '\n')
            result_file.write('Average number of simulations per iteration: ' + str(av_total_simulations) + '\n\n\n')


def write_results(results, result_directory):
    for model_name, model_result_list in results.items():
        for alpha_value in model_result_list:
            model_result_list[alpha_value].sort(key=result_comparator)
            write_model_results(model_name, model_result_list[alpha_value], alpha_value, result_directory)

def plot_alpha_graphs(results, result_directory):
    plotsDir = os.path.join(result_directory, "plots")
    if not os.path.isdir(plotsDir):
        os.makedirs(plotsDir)
    for model_name, model_result_list in results.items():

        for iterSample in iterSamples:
            xValues = []
            totalSamplesValues = []
            totalSimulationsValues = []

            for alpha_value in alphaValues:
                xValues.append(alpha_value)
                filteredOutput = [y for y in model_result_list[alpha_value] if y.iter_sample == iterSample]
                totalSamplesValues.append(sum([y.total_samples for y in filteredOutput]) / len(filteredOutput))
                totalSimulationsValues.append(sum([y.total_simulations for y in filteredOutput]) / len(filteredOutput))

    #        yValues = [sum([y.total_samples for y in x])/ len(x) for x in list(model_result_list.values())]
            print(xValues)
            print(totalSamplesValues)
            plt.plot(xValues, totalSamplesValues, "C1", label="total samples")
            plt.legend()
            plt.xlabel("alpha")
            plt.ylabel("Total Actions Sampled")
            samples_filename = f"{model_name.replace('.', '-')}-samples-{iterSample}.png"
            plt.title(f"{model_name} {iterSample}")
            print("Saving graphs at")
            print(os.path.join(plotsDir, samples_filename))
            plt.savefig(os.path.join(plotsDir, samples_filename))
            plt.close()

            print(xValues)
            print(totalSimulationsValues)
            plt.plot(xValues, totalSimulationsValues, "C2", label="total simulations")
            plt.legend()
            plt.xlabel("alpha")
            plt.ylabel("Total simulations")
            simulations_filename = f"{model_name.replace('.', '-')}-simulations-{iterSample}.png"
            plt.title(f"{model_name} {iterSample}")
            print("Saving graphs at")
            print(os.path.join(plotsDir, simulations_filename))
            plt.savefig(os.path.join(plotsDir, simulations_filename))
            plt.close()


def remove_old_results():
    is_dir_exists = os.path.isdir(resultDirectory)
    if is_dir_exists:
        shutil.rmtree(resultDirectory)


# Handling input
parser = argparse.ArgumentParser()
inputOptions.add_basic_input_options(parser)
inputOptions.add_n_experiments_option(parser)
arguments = parser.parse_args()
input_values = inputOptions.parse_user_input(arguments)
number_of_experiments = arguments.nExperiments


resultDirectory = input_values.output_directory + '/'
remove_old_results()
exec_command = get_exec_command_from_input()
number_of_threads_in_exec = input_values.number_of_threads
schedule_and_run_benchmarks(number_of_experiments)
benchmarkInfo = benchmarksUtil.accumulate_results(resultDirectory)
write_results(benchmarkInfo, resultDirectory)
print("writing results in " + resultDirectory)
plot_alpha_graphs(benchmarkInfo, resultDirectory)