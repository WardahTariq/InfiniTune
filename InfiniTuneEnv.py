
import numpy as np
import copy
from fabric import task, Connection
import json
from multiprocessing import Process
import time
from Parser import Parser  # Assuming Parser is a custom module in your project
import subprocess

# Load configuration
with open('driver_config.json', 'r') as f:
    CONF = json.load(f)

def run_command(command):
    # Hardcoded filename
    filename = './tmpLog/output.txt'

    # Check if the command requires sudo
    if command.startswith('sudo'):
        # Prompt for the sudo password
        sudo_password = '15032002'

        # Create a command list to pass to subprocess.run
        command_list = command.split()
        # print(f"command_list = {command_list}")

        # Execute the command and capture the output
        result = subprocess.run(command_list, input=sudo_password+'\n', shell=False,
                                capture_output=True, text=True)
    else:
        # Execute the command and capture the output
        print(f"command {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        print("Command executed successfully.")
        # Print the output of the command
        print("Output:")
        print(result.stdout)
        
        # Write the output to the hardcoded filename, overwriting existing content
        with open(filename, 'w') as f:
            f.write(result.stdout)
    else:
        print("Command failed.")
        # Print the error output
        print("Error:")
        print(result.stderr)


def free_cache():
    run_command('sync')
    run_command('sudo purge')


def run_benchmark(workload=None):
    # c.run('PGPASSWORD=bohan psql --cluster 10/main -U bohan -d tpch40 -a -f ./q13.sql -o tpch.log > ./query.log')
    if not workload:
        run_command('pgbench -c 10 -j 10 -t 10000 random')
        return
    command=f'pgbench -c {workload[1]} -j {workload[2]} -t {workload[3]} {workload[0]}'
    # run_command(f'pgbench -i -s {workload[4]}')
    # time.sleep(10)
    run_command(command)
    # time.sleep(10)


def get_latency():
    
    with open('./tmpLog/output.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('latency average'):
                # Split the line and extract the latency value
                latency_value = line.split('=')[1].strip().split()[0]
                return latency_value
    return None

# @task
def restart_database():
    run_command('brew services restart postgresql@16')
    # c.sudo('service postgresql restart')

# @task
def save_dbms_result(c):
    t = int(time.time())
    files = ['knobs.json', 'metrics_after.json', 'metrics_before.json', 'summary.json']
    for f_ in files:
        f_prefix = f_.split('.')[0]
        c.run('cp ../controller/output/{} {}/{}__{}.json'.format(f_, CONF['save_path'], t, f_prefix))

    c.run('cp tpch.log {}/{}__{}.txt'.format(CONF['save_path'], t, 'queryplan'))

@task
def upload_result(c):
    c.run('python3 ../../server/website/script/upload/upload.py ../controller/output/ {} {}/new_result/'.format(CONF['upload_code'], CONF['upload_url']))

@task
def add_udf(c):
    c.sudo('python3 ../driver/LatencyUDF.py ../controller/output/ {}'.format('tpch.log'))

class InfiniTuneEnv():
    def __init__(self, min_vals, max_vals, default_vals, knob_names):
        self.min_vals = np.array(min_vals)
        self.max_vals = np.array(max_vals)
        self.default_vals = np.array(default_vals)
        self.knob_names = np.array(knob_names)
        self.N = len(knob_names)
        self.knob_id = 0
        assert(self.N > 0)
        assert(len(min_vals) == self.N)
        assert(len(max_vals) == self.N)

    def reset(self):
        self.knob_id = 0
        # initial state is default config
        initial_state = np.zeros(self.N + 1)
        scaled_vals = Parser().scaled(self.min_vals, self.max_vals, self.default_vals)
        initial_state[:self.N] = scaled_vals
        return initial_state

    def step(self, action, state):
        knob_id = self.knob_id
        nextstate = copy.copy(state)
        print(f"knob_id {knob_id}")
        nextstate[knob_id] = action
        nextstate[self.N] = knob_id + 1
        debug_info = {}
        reward = 0
        if knob_id < self.N - 1 and knob_id >= 0:
            is_terminal = False
        elif knob_id == self.N - 1:
            is_terminal = True
        else:
            raise Exception("Invalid Knob ID {}. ".format(knob_id))

        self.knob_id += 1
        return (nextstate, reward, is_terminal, debug_info)

    def run_experiment(self,workload=None):
        # connect_kwargs={'key_filename': '/Users/warda/.ssh/ddpg'}
        # with Connection('localhost',user='warda', port=22, connect_kwargs=connect_kwargs) as c:
        # free cache
        free_cache()

        # restart database
        restart_database()

        # Start controller in a separate process
        # p = Process(target=run_controller, args=(c,))
        # p.start()

        time.sleep(15)

        # Run TPCH
        run_benchmark(workload)

        # Stop controller
        # stop_controller(c)
        # p.join()

        reward = float(get_latency())
        print(reward)
        return reward

    def change_conf(self, config_vals):
        conf_path = '/opt/homebrew/var/postgresql@16/postgresql.conf' 
        with open(conf_path, "r+") as postgresqlconf:
            lines = postgresqlconf.readlines()
            settings_idx = lines.index("# Add settings for extensions here\n")
            postgresqlconf.seek(0)
            postgresqlconf.truncate(0)

            lines = lines[0:(settings_idx + 1)]
            for line in lines:
                postgresqlconf.write(line)

            for i in range(len(self.knob_names)):
                s = str(self.knob_names[i]) + ' = ' + str(config_vals[i]) + "\n"
                print (s)
                postgresqlconf.write(s)
                
        # conf_path = '/opt/homebrew/var/postgresql@16/postgresql.conf'
        # with Connection('localhost') as c:
        #     for i, knob_name in enumerate(self.knob_names):
        #         s = '{} = {}\n'.format(knob_name, config_vals[i])
        #         c.sudo("sed -i '/{}/c\\{}' {}".format(knob_name, s, conf_path))

    def save_and_upload(self):
        pass
        # add_udf()
        # save_dbms_result()
        # upload_result()