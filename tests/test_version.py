from dizher.version import Build


def test_display():
    assert Build('0.2.0', 'alpha', 'abc1234').display == '0.2.0-alpha'
    assert Build('0.2.0', 'alpha', 'abc1234', dirty=True).display == '0.2.0-alpha-dirty'
    assert Build('0.2.0', 'dev', 'abc1234').display == '0.2.0-dev+gabc1234'
    assert Build('0.2.0', 'dev', 'abc1234', dirty=True).display == '0.2.0-dev+gabc1234-dirty'
    assert Build('0.2.0', 'dev').display == '0.2.0-dev'   # outside git
