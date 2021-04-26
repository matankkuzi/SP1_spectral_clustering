# tasks module
#
# This module allows the user to run
# the program via invoke commands.

from invoke import task


@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(build)
def run(c, k=0, n=0, Random=True):
    if Random:
        # default k and n in case of Random
        k=2
        n=10
        c.run("python3.8.5 main.py {} {} {}".format(k, n, "--Random"))
    else:
        # in case that noRandom and k or n is missing main.py flags
        c.run("python3.8.5 main.py {} {} {}".format(k, n, "--no-Random"))


@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")
