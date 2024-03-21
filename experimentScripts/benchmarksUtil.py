from maxRewards import get_max_reward
import os

class ModelResult:
    def __init__(self, model_name, times, lower_bounds, upper_bounds, num_explored_states, missing_probability, iteration_number, total_samples, alpha):
        self.model_name = model_name
        self.times = times
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.num_explored_states = num_explored_states
        self.missing_probability = missing_probability
        self.iteration_number = iteration_number
        self.total_samples = total_samples
        self.alpha = alpha

    def get_runtime(self):
        time_taken_millis = self.times[-1] - self.times[0]
        return time_taken_millis/1000

    def get_bounds(self):
        return self.lower_bounds[-1], self.upper_bounds[-1]

    def get_bounds_diff(self):
        return self.upper_bounds[-1] - self.lower_bounds[-1]


def parse_output_file(file_name, iteration_number=0):
    file_stream = open(file_name)
    content = file_stream.readlines()
    file_stream.close()

    options = content[0].split(" -")
    model_name = options[0].split("/")[-1].split(".prism")[0]
    content = content[1:]
    times = list(map(float, content[0].split()))
    lower_bound = list(map(float, content[1].split()))
    scaled_lower_bounds = [x/get_max_reward(model_name) for x in lower_bound]
    upper_bound = list(map(float, content[2].split()))
    scaled_upper_bounds = [x/get_max_reward(model_name) for x in upper_bound]
    explored_states = int(content[3])
    total_samples = int(content[4])
    alpha = float(content[5])

    missing_probability = None
    if len(content) > 5:
        missing_probability = float(content[5])

    return ModelResult(model_name, times, scaled_lower_bounds, scaled_upper_bounds, explored_states, missing_probability, iteration_number, total_samples, alpha)


def accumulate_results(resultDirectory):
    # Now store all the information in variables
    benchmark_info = {}
    num_iterations = 0
    for directory in os.listdir(resultDirectory):
        if not os.path.isdir(os.path.join(resultDirectory, directory)):
            continue

        files = os.listdir(os.path.join(resultDirectory, directory))

        for file in files:
            if not file.split(".")[0].isnumeric():
                continue

            file_path = os.path.join(resultDirectory, directory, file)
            model_result = parse_output_file(file_path, num_iterations)

            if model_result.model_name not in benchmark_info:
                benchmark_info[model_result.model_name] = {}

            if model_result.alpha not in benchmark_info[model_result.model_name]:
                benchmark_info[model_result.model_name][model_result.alpha] = []

            benchmark_info[model_result.model_name][model_result.alpha].append(model_result)

        num_iterations = num_iterations + 1

    return benchmark_info


def get_average_values(model_result_list):
    average_lower_bound = 0
    average_upper_bound = 0
    average_run_time = 0
    average_states_explored = 0
    average_total_samples = 0
    for model_result in model_result_list:
        average_lower_bound += model_result.lower_bounds[-1]
        average_upper_bound += model_result.upper_bounds[-1]
        average_run_time += model_result.get_runtime()
        average_states_explored += model_result.num_explored_states
        average_total_samples += model_result.total_samples

    average_lower_bound /= len(model_result_list)
    average_upper_bound /= len(model_result_list)
    average_run_time /= len(model_result_list)
    average_states_explored /= len(model_result_list)
    average_total_samples /= len(model_result_list)

    return average_lower_bound, average_upper_bound, average_run_time, average_states_explored, average_total_samples
