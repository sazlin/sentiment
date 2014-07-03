from skeleton import _populate_vocab_dict
import os

test_pos_text = """moviemaking is a lot like being the general manager of an nfl team in the post-salary cap era -- you've got to know how to allocate your resources .
every dollar spent on a free-agent defensive tackle is one less dollar than you can spend on linebackers or safeties or centers .
in the nfl , this leads to teams like the detroit lions , who boast a superstar running back with a huge contract , but can only field five guys named herb to block for him .
in the movies , you end up with films like " spawn " , with a huge special-effects budget but not enough money to hire any recognizable actors .
jackie chan is the barry sanders of moviemaking . """


def test_populate_vocab_dict():
    test_stopwords = set(['teststop', 'dollar'])
    f = open('./test_files/test_file', 'w')
    f.write(test_pos_text)
    f.close()
    test_d = {}
    _populate_vocab_dict(test_d, test_stopwords, ['test_file'], 'test_files')
    assert "moviemaking" in test_d
    assert test_d["moviemaking"] == 2
    assert "dollar" not in test_d
    assert "superrandomword" not in test_d
