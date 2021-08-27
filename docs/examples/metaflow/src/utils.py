import functools


def pip(libraries, test_index=False):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import subprocess
            import sys

            for library, version in libraries.items():
                if not test_index:
                    print("Pip Install:", library, version)
                    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", library + "==" + version])
                else:
                    print("Pip Test Install:", library, version)

                    if version.startswith('git+https://') or version.startswith('https://'):
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                "--quiet",
                                version,
                            ]
                        )
                    else:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                "--quiet",
                                "--index-url",
                                "https://test.pypi.org/simple/",
                                "--extra-index-url",
                                "https://pypi.org/simple",
                                library + "==" + version,
                            ]
                        )
            return function(*args, **kwargs)

        return wrapper

    return decorator