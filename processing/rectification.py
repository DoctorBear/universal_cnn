import re
from typing import Generator


def rectify_by_location(char_gen: Generator):
    for char_list in char_gen:
        char = char_list[0]
        if char.c in ('，', '’'):
            char.set_content_text('，' if char.location() == 'floor' else '’',
                                  msg='char is on %s' % char.location())


def rectify_5(char_gen: Generator):
    """
    `‘`+`‘` = `“`
    `’`+`’` = `”`
    :param char_gen:
    :return:
    """
    num_ptn = re.compile('\d')
    letter_ptn = re.compile('[a-zA-Z]')
    for char_5 in char_gen:
        lefter, left, char, right, righter = char_5
        if char.c == right.c == '‘':
            char.set_content_text('“', msg='`‘` on right')
            right.set_content_text('', msg='merged into left')
        elif char.c == right.c == '’':
            char.set_content_text('”', msg='`’` on right')
            right.set_content_text('', msg='merged into left')
        elif char.c == 'O':
            num_score = sum(map(lambda _: bool(num_ptn.match(_.c)), [lefter, left, right, righter]))
            letter_score = sum(map(lambda _: bool(letter_ptn.match(_.c)), [lefter, left, right, righter]))
            if num_score > letter_score:
                char.set_content_text('0', msg='more numbers around %d>%d' % (num_score, letter_score))
        elif char.c == '0':
            num_score = sum(map(lambda _: bool(num_ptn.match(_.c)), [lefter, left, right, righter]))
            letter_score = sum(map(lambda _: bool(letter_ptn.match(_.c)), [lefter, left, right, righter]))
            if letter_score > num_score:
                char.set_content_text('O', msg='more letters around %d>%d' % (letter_score, num_score))
        
        if char.c in ('I', 'l'):
            if right.c in ('、',):
                char.set_content_text('1', msg='`、` on right')
            else:
                num_score = sum(map(lambda _: bool(num_ptn.match(_.c)), [lefter, left, right, righter]))
                letter_score = sum(map(lambda _: bool(letter_ptn.match(_.c)), [lefter, left, right, righter]))
                if num_score > letter_score:
                    char.set_content_text('1', msg='more numbers around %d>%d' % (num_score, letter_score))