from rh2.sdk.env import get_rh2_env

from rh2_jobs.rh2_utils import Rh2ConvertingParamsParser
from rh2_jobs.rh2_main import converter_main


def main():
    rh2_env = get_rh2_env()
    converter_main(exec_path=__file__,
                   rh2_env=rh2_env,
                   rh2_params_parser=Rh2ConvertingParamsParser,
                   cfg_yaml=rh2_env.params.converting_yaml)


if __name__ == '__main__':
    main()
