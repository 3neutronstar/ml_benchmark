import json,os
def load_params(configs, file_name):
    ''' replay_name from flags.replay_name '''
    with open(os.path.join(configs['current_path'], 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs

    
def save_params(configs, time_data):
    with open(os.path.join(configs['current_path'], 'grad_data', '{}.json'.format(time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)
