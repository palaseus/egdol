import json
import subprocess
import sys


def run(cmd):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode(), p.stderr.decode()


def test_script_human(tmp_path):
    script = tmp_path / 's.egdol'
    script.write_text('fact: a(b).\n? a(b).\n')
    code, out, err = run(f"{sys.executable} -m egdol.main --script {script}")
    assert code == 0
    assert 'true.' in out


def test_script_json(tmp_path):
    script = tmp_path / 's2.egdol'
    script.write_text('fact: a(b).\n? a(b).\n')
    code, out, err = run(f"{sys.executable} -m egdol.main --script {script} --output-format json")
    assert code == 0
    data = json.loads(out)
    assert 'results' in data
 