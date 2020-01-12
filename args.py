import yaml
import frozen_dir


f = open(frozen_dir.app_path()+'configs/single_char_3991.yaml', encoding='utf-8')
# f = open('configs/punctuation_letter_digit.yaml', encoding='utf-8')
args = yaml.load(f.read())
f.close()
