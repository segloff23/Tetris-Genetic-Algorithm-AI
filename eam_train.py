
import time
import os
import argparse
import sys
sys.path.append('.') # Seems to prevent unkown issue with multiproccessing

import dill as pickle
from pathlib import Path
from numpy.random import seed

from tetris import Tetris
from heuristics import Heuristics
from ecosystem import Ecosystem

###############################################################################

save_load_folder = 'ecosystems'
log_folder = 'logs'
save_load_file_path = Path('./' + save_load_folder + '/')
log_file_path = Path('./' + log_folder + '/')

def printf(message, toggle_off):
    """Modified print() to help toggle messages on/off.

    Parameters
    ----------
    message : (str)
        Message to be printed.
    toggle_off : (bool)
        Whether the message should not be printed to console.

    Returns
    -------
    None.

    """
    if not toggle_off:
        print(message)

def load_config_settings(file_name, suppress_warnings=False):
    """Loads settings for eam_train.py from given file.

    Parameters
    ----------
    file_name : (str)
        Name of text file containing settings. Must include '.txt'.
    suppress_warnings : (bool), optional
        Do not print warning messages to console. Errors which prevent
        the run from occuring are still displayed.
        The default is False.

    Returns
    -------
    valid_input : (bool)
        Truth value of whether the input is valid.
    inputs : (dict)
        Set of name, value pairs describing the setting configuration.

    """

    # NAME : DEFAULT [or required if no default exists]
    keywords = {'WIDTH' : 6,
                'HEIGHT' : 12,
                'MAX_SPAWNS' : None,
                'ACTIVE_TETROMINOS' : None,
                'MODEL_NAME' : 'required',
                'GEN_SIZE' : 100,
                'GAME_SIZE' : 5,
                'MUT_PROB' : 0.1,
                'N_EPISODES' : 100,
                'EXP' : 'required',
                'SAVE_FREQ' : 300,
                'USE_POOL' : True,
                'SEED' : None,
                'LOGGING' : False}

    # NAME : (type [,type], None acceptable)
    keyword_types = {'WIDTH' : (int, False), 'HEIGHT' : (int, False),
                     'MAX_SPAWNS' : (int, True),
                     'ACTIVE_TETROMINOS' : (list, int, True),
                     'MODEL_NAME' : (str, False), 'GEN_SIZE' : (int, False),
                     'GAME_SIZE' : (int, False), 'MUT_PROB' : (float, False),
                     'N_EPISODES' : (int, False), 'EXP' : (int, False),
                     'SAVE_FREQ' : (int, False), 'USE_POOL' : (bool, False),
                     'SEED' : (int, True), 'LOGGING' : (bool, False)}

    invalid_string = 'Warning: Invalid value encountered with key {}.'
    default_string = 'Warning: Using default value {} for {}.'
    required_string = 'Error: {} is a required parameter.'
    unrecognized_string = 'Warning: {} not a recognized key.'

    inputs = {key : 'empty' for key in keywords}
    with open(file_name, 'r') as file:

        valid_input = True
        for line in file.readlines():
            clean_line = line.strip()
            if len(clean_line) != 0 and clean_line[0] != '#':
                key, raw_value = line.strip().split('=')
                key = key.strip().upper()
                raw_value = raw_value.strip()
                if key in keywords:
                    if raw_value.lower() == 'none':
                        inputs[key] = None
                    elif keyword_types[key][0] == bool:
                        value = raw_value.lower()
                        if value == 'true':
                            inputs[key] = True
                        elif value == 'false':
                            inputs[key] = False
                        else:
                            printf(invalid_string.format(key),
                                   suppress_warnings)
                    elif keyword_types[key][0] == list:
                        try:
                            inputs[key] = [keyword_types[key][1](x) for x in raw_value.split(',')]
                        except:
                            printf(invalid_string.format(key),
                                   suppress_warnings)
                    else:
                        try:
                            inputs[key] = keyword_types[key][0](raw_value)
                        except:
                            printf(invalid_string.format(key),
                                   suppress_warnings)
                elif not suppress_warnings:
                    printf(unrecognized_string.format(key),
                                   suppress_warnings)

    valid_input = True
    for key in inputs:
        if inputs[key] == 'empty':
            if keywords[key] != 'required':
                inputs[key] = keywords[key]
                if not suppress_warnings:
                    printf(default_string.format(keywords[key], key),
                                   suppress_warnings)
            else:
                valid_input = False
                printf(required_string.format(key),
                                   suppress_warnings)
        elif inputs[key] == None:
            if not keyword_types[key][-1]:
                if keywords[key] != 'required':
                    inputs[key] = keywords[key]
                    if not suppress_warnings:
                        printf(default_string.format(keywords[key], key),
                                   suppress_warnings)
                else:
                    valid_input = False
                    printf(required_string.format(key),
                                   suppress_warnings)

    return valid_input, inputs

def load_ecosystem(EXP, quiet):
    """Loads an ecosystem from a given file

    Returns
    -------
    ecosystem : (ecosystem.Ecosystem)
        The ecosystem class which evolution will take place in.
    """

    file_name = save_load_file_path / ('exp' + str(EXP))

    with open(file_name, 'rb') as inputs:
        ecosystem = pickle.load(inputs)
    printf('Ecosystem created.', quiet)

    return ecosystem

def validate_save_file(EXP, quiet):
    """Verifies the user wants to use the given experiment number.

    Returns
    -------
    bool
        True to continue evolution, False to cancel.

    """

    experiments = [int(f.name[3::]) for f in os.scandir(save_load_file_path)
                   if 'config' not in f.name]
    if EXP in experiments:
        s = 'Warning: An ecosystem exists for experiment #{}.\n'
        print(s.format(EXP) +'This file will be overwritten.')

        while True:
            response = input('Continue? (y/n) : ')
            if response == 'n':
                return False
            elif response == 'y':
                printf('', quiet)
                return True
            else:
                print('Invalid input. Please try again.', end='')
    else:
        return True

def main(argv):
    """Evolve an ecosystem to play Tetris with the given constant parameters.
    """

    config_help = 'path of file for training configuration, include .txt'
    warnings_help = ('supress warning messages during configuration file'
                     + 'processing')
    quiet_help = ('supress console output during training, '
                  + 'except critical errors')
    load_help = ('load an experiment given by its number, and use the saved '
                 + 'configuration or the one specified by --config')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help=config_help)
    parser.add_argument('-l', '--load', type=int, help=load_help)
    parser.add_argument('-w', '--warnings', action='store_true',
                        help=warnings_help)
    parser.add_argument('-q', '--quiet', action='store_true',
                        help=quiet_help)
    args = vars(parser.parse_args())

    quiet = args['quiet']
    
    if args['load'] is not None:
        if args['config'] is None:
            args['config'] = save_load_file_path / ('config_exp'
                                                    + str(args['load']))
    else:    
        if args['config'] is None:
            args['config'] = Path('./') / 'default_config.txt'

    valid, settings = load_config_settings(args['config'],
                                           suppress_warnings=args['warnings'])

    if not valid:
        print('Unable to solve configuration file. Exiting.')
        sys.exit(1)

    seed(settings['SEED'])
    env = Tetris(width=settings['WIDTH'], height=settings['HEIGHT'],
                 max_spawns=settings['MAX_SPAWNS'],
                 active_tetrominos=settings['ACTIVE_TETROMINOS'])
    heuristics = Heuristics(env)

    printf('', quiet)
    if args['load'] is not None:
        ecosystem = load_ecosystem(settings['EXP'], quiet)
    else:
        valid_save = validate_save_file(settings['EXP'], quiet)
        if not valid_save:
            return
        else:
            from importlib import import_module
            from ecosystem import generate_model

            with open(args['config'], 'r') as file:       
                config_data = file.readlines()
            
            config_save_name = 'config_exp' + str(settings['EXP'])
            with open(save_load_file_path / (config_save_name), 'w') as file:
                file.write(''.join(config_data))

            model_generator = import_module('models.'
                                            + settings['MODEL_NAME'])
            model_settings, evaluator = model_generator.get_model_settings()
            model = generate_model(*model_settings)

            ecosystem = Ecosystem(env, heuristics, settings['GEN_SIZE'],
                                  model, evaluator,
                                  settings['MODEL_NAME'])
            printf('Ecosystem created.', quiet)

    if settings['LOGGING']:
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            log_folder_name = log_file_path / ('exp' + str(settings['EXP']))
            writer = SummaryWriter(log_folder_name)
            disable_logging = False
        except:
            printf('torch.utils.tensorboard not found, logging disabled.',
                   args['warnings'])
            disable_logging = True

    if settings['USE_POOL']:
        from multiprocessing import Pool, cpu_count
        pool = Pool(cpu_count())
    else:
        pool = None

    save_file_name = save_load_file_path / ('exp' + str(settings['EXP']))

    s ='Saving to /' + save_load_folder + '/exp{} every {} seconds.'
    printf(s.format(settings['EXP'], settings['SAVE_FREQ']), quiet)
    printf('Beginning evolution.', quiet)
    printf('', quiet)

    report_string = 'Episode {}: {:7.2f} / {:<7.2f}'
    start = time.time()
    for ep in range(settings['N_EPISODES']):

        mean_score, elite_mean_score = ecosystem.evolve(settings['GAME_SIZE'],
                                                        settings['GEN_SIZE'],
                                                        settings['MUT_PROB'],
                                                        pool=pool,
                                                        quiet=quiet)

        if settings['LOGGING'] and not disable_logging:
            writer.add_scalar('Mean_Score', mean_score, ep+1)
            writer.add_scalar('Elite_Mean_Score', elite_mean_score, ep+1)

        printf(report_string.format(ecosystem.generation_number,
                                   mean_score, elite_mean_score), quiet)

        if (ep == 0) or (time.time() - start) >= settings['SAVE_FREQ']:
            with open(save_file_name, 'wb') as output:
                pickle.dump(ecosystem, output, -1)
            printf('Ecosystem saved.', quiet)
            start = time.time()
        printf('', quiet)

if __name__ == '__main__':
    main(sys.argv[1::])

#EOF