# coding: utf-8

import os
import shlex
import shutil
import subprocess
import sys

from .logger import logger

if sys.version_info[0] >= 3:
    _STRING_TYPES = (str,)
else:
    raise ImportError(f'python{sys.version_info[0]} is not supported.')

# supported HDFS FS prefix
_SUPPORTED_HDFS_PATH_PREFIXES = ('hdfs://', 'ufs://')


__all__ = [
    'has_hdfs_path_prefix',
    'is_hdfs_path_pattern',
    'exists_or_islink',
    'check_call_hdfs_command',
    'popen_hdfs_command',
    'is_hdfs_file',
    'is_hdfs_dir',
    'hdfs_mkdir',
    'hdfs_put',
    'hdfs_get',
    '_hdfs_get',
    '_hdfs_ls',
    'hdfs_ls',
    'get_bytesnake_path',
    'hdfs_rm',
    'download_from_hdfs'
]


def has_hdfs_path_prefix(filepath):
    """
    Check if a filepath has hdfs path prefix.

    Args:
        filepath: str, filepath.
    Return: bool, if a filepath has hdfs path prefix.
    """
    for prefix in _SUPPORTED_HDFS_PATH_PREFIXES:
        if filepath.startswith(prefix):
            return True
    return False


def is_hdfs_path_pattern(filepath):
    """
    Check if a filepath is a hdfs path pattern.

    Args:
        filepath: str, filepath.
    Return: bool, if a filepath is of hdfs path pattern.
    """
    return filepath.find('*') != -1 or \
        (filepath.find('[') != -1 and filepath.find(']') != -1)


def exists_or_islink(filepath):
    """
    Check file exists or not, allow symbol links.

    Args:
        filepath: str, filepath.
    Return: bool, if a file exists or is a symbol link.
    """
    return os.path.exists(filepath) or os.path.islink(filepath)


def check_call_hdfs_command(command, hadoop_binary='hadoop'):
    """
    Check call hdfs command with hadoop_binary.

    Args:
        command: str, hadoop command.
        hadoop_binary: str, hadoop binary.
    Return: None
    """
    hdfs_command = '{} fs {}'.format(hadoop_binary, command)
    subprocess.check_call(shlex.split(hdfs_command))


def popen_hdfs_command(command, hadoop_binary='hadoop'):
    """
    Call hdfs command with popen and get stdout result.

    Args:
        command: str, hadoop command.
        hadoop_binary: str, hadoop binary.
    Return: stdout result.
    """
    hdfs_command = '{} fs {}'.format(hadoop_binary, command)
    p = subprocess.Popen(shlex.split(hdfs_command), stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    return stdout


def is_hdfs_file(filepath, hadoop_binary='hadoop'):
    """
    Check if input filepath is hdfs file.

    Args:
        filepath: str, filepath.
        hadoop_binary: str, hadoop binary.
    Return: bool, if input filepath is hdfs file.
    """
    if exists_or_islink(filepath):
        # is local path, return False
        return False
    elif is_hdfs_path_pattern(filepath):
        # this is a hdfs path pattern, not a file
        return False
    cmd = '-test -f {}'.format(filepath)
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        return False


def is_hdfs_dir(filepath, hadoop_binary='hadoop'):
    """
    Check if input filepath is hdfs directory.

    Args:
        filepath: str, filepath.
        hadoop_binary: str, hadoop binary.
    Return: bool, if input filepath is hdfs directory.
    """
    if exists_or_islink(filepath):
        # is local path, return False
        return False
    elif is_hdfs_path_pattern(filepath):
        # this is a hdfs path pattern, not a directory
        return False
    cmd = '-test -d {}'.format(filepath)
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception:
        return False


def hdfs_mkdir(filepath, hadoop_binary='hadoop', raise_exception=False):
    """
    Make hdfs directory.

    Args:
        filepath: str, filepath.
        hadoop_binary: str, hadoop binary.
        raise_exception: bool, whether to raise exception.
    Return: if making directory successes.
    """
    try:
        cmd = '-mkdir -p {}'.format(filepath)
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception as e:
        msg = 'Failed to mkdir {} in HDFS: {}'.format(filepath, e)
        if raise_exception:
            raise ValueError(msg)
        else:
            logger.error(msg)
        return False


def hdfs_put(
        src,
        dst,
        overwrite=False,
        output_to_dir=False,
        hadoop_binary='hadoop'):
    """
    Upload src files/directories to dst path.

    Args:
        src: (str, List(str), Tuple(str)), source of downloading.
        dst: str, destination of uploading.
        overwrite: bool, whether overwrite exist files or not.
        output_to_dir: bool, if dst is a dir.
            will be set True if src is a (list, tuple) with more
            than one element.
        hadoop_binary: str, hadoop binary.
    Return: bool, whether uploading successes.
    """
    require_dst_dir = True if output_to_dir else False
    assert isinstance(src, (list, tuple) + _STRING_TYPES), \
        f"Input src path must be a str or a list of str, got {src}"
    assert src, "Input src path is empty"
    if isinstance(src, (list, tuple)):
        if len(src) > 1:
            # dst path must be a directory
            require_dst_dir = True
    else:
        src = [src]

    # check output dst path
    if not has_hdfs_path_prefix(dst):
        raise ValueError(f'Input dst path is not a hdfs path: {dst}')
    if require_dst_dir:
        if is_hdfs_file(dst):
            raise IOError(
                f'Required dst path {dst} as a directory for uploading, '
                f'got a file')
        elif not is_hdfs_dir(dst):
            # mkdir
            hdfs_mkdir(dst)
    else:
        dst_dir = os.path.dirname(dst)
        if not is_hdfs_dir(dst_dir):
            # mkdir
            if is_hdfs_path_pattern(dst_dir):
                # this is a hdfs path pattern, cannot make this directory
                raise OSError(
                    f"HDFS destination directory cannot be a wildcard "
                    f"pattern, got {dst_dir}")
            hdfs_mkdir(dst_dir)

    hdfs_cmd = '-put -f' if overwrite else '-put'

    cmd = f"{hdfs_cmd} {' '.join(src)} {dst}"
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception as e:
        logger.error(
            f'HDFS put command exception caught: '
            f'type {type(Exception).__name__}, msg: {e}')
        return False


def hdfs_get(src, dst, cli, output_to_dir=False):
    """
    Download src files/directories to dst path.

    Args:
        src: (str, List(str), Tuple(str)), source of downloading.
        dst: str, destination of downloading.
        cli: bytesnake HDFS cli.
        output_to_dir: bool, if dst is a dir.
            will be set True if src is a (list, tuple) with more
            than one element.
    Return: bool, whether downloading successes.
    """
    require_dst_dir = True if output_to_dir else False
    assert isinstance(src, (list, tuple) + _STRING_TYPES), \
        f"Input src path must be a str or a list of str, got {type(src)}"
    assert src, "Input src path is empty"
    if isinstance(src, (list, tuple)):
        if len(src) > 1:
            # dst path must be a directory
            require_dst_dir = True
    else:
        src = [src]

    # check output dst path
    if require_dst_dir:
        if os.path.exists(dst) and os.path.isfile(dst):
            raise IOError(
                f'Required dst path {dst} as a directory for multiple '
                f'hdfs paths downloading, got a file')
        elif not os.path.exists(dst):
            os.makedirs(dst)
        dst_dir = dst
    else:
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

    src = [get_bytesnake_path(src_i) for src_i in src]
    try:
        cat_generators = cli.cat(src)
        for i, str_i in enumerate(cat_generators):
            if require_dst_dir:
                in_filename = os.path.basename(src[i])
                out_filepath = os.path.join(dst_dir, in_filename)
            else:
                out_filepath = dst
            with open(out_filepath, 'w') as f:
                # str_i is still a wrapper.
                f.write(str_i.next())
        return True
    except Exception as e:
        logger.error(
            f'HDFS get command exception caught: '
            f'type {type(e).__name__}, msg: {e}')
        return False


def _hdfs_get(
        src,
        dst,
        overwrite=False,
        output_to_dir=False,
        hadoop_binary='hadoop'):
    """
    Download src files/directories to dst path.

    Args:
        src: (str, List(str), Tuple(str)), source of downloading.
        dst: str, destination of downloading.
        overwrite: bool. If True, the local file will be overwritten
            if it exists.
        output_to_dir: bool, if dst is a dir.
            will be set True if src is a (list, tuple) with more
            than one element.
        hadoop_binary: str, hadoop binary.
    Return: bool, whether downloading successes.
    """
    require_dst_dir = True if output_to_dir else False
    assert isinstance(src, (list, tuple) + _STRING_TYPES), \
        f"Input src path must be a str or a list of str, got {src}"
    assert src, "Input src path is empty"
    if isinstance(src, (list, tuple)):
        if len(src) > 1:
            # dst path must be a directory
            require_dst_dir = True
    else:
        src = [src]

    # check overwirte option
    if overwrite is True:  # Remove the targeted file/folder if it exists
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.isfile(dst):
            os.remove(dst)
    else:  # skip downloading if the targeted file/folder exists
        if os.path.exists(dst):
            return True

    # check output dst path
    if require_dst_dir:
        if os.path.exists(dst) and os.path.isfile(dst):
            raise IOError(
                f'Required dst path {dst} as a directory for multiple '
                f'hdfs paths downloading, got a file')
        elif not os.path.exists(dst):
            os.makedirs(dst)
    else:
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)

    hdfs_cmd = f"-get {' '.join(src)} {dst}"
    try:
        check_call_hdfs_command(hdfs_cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception as e:
        logger.error(
            f'HDFS get command exception caught: '
            f'type {type(e).__name__}, msg: {e}')
        return False


def _hdfs_ls(filepath, hadoop_binary='hadoop'):
    """
    List hdfs path pattern.

    Args:
        filepath: str, filepath.
        hadoop_binary: str, hadoop binary.
    Return: List(str), the result of listing.
    """
    try:
        cmd = '-ls {}'.format(filepath)
        stdout = popen_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        lines = stdout.splitlines()
        if lines:
            # decode bytes string in python3 runtime
            lines = [line.decode('utf-8') for line in lines]
            return [line.split(' ')[-1] for line in lines][1:]
        else:
            return []
    except Exception:
        return []


def hdfs_ls(filepath, cli):
    """
    List hdfs path pattern by bytesnake.

    Args:
        filepath: str, filepath.
        cli: bytesnake HDFS cli.
    Return: List(str), the result of listing.
    """
    bytesnake_path = get_bytesnake_path(filepath)
    return list(cli.ls([bytesnake_path]))


def get_bytesnake_path(filepath):
    """
    Convert hdfs:// path to bytesnake path.

    Args:
        filepath: str, filepath.
    Return: str, bytesnake path.
    """
    return '/' + '/'.join(filepath.split('/')[3:])


def hdfs_rm(filepath, recursive=True, force=True, hadoop_binary='hadoop'):
    """
    Remove files from hdfs filepath.

    Args:
        filepath: str, filepath.
        recursive: bool, whether do recursive removal.
        force: bool, whether do force removal.
        hadoop_binary: str, hadoop binary.
    Return: bool, whether removing successes.
    """
    assert isinstance(filepath, _STRING_TYPES), \
        f"Input filepath must be a str, got {type(filepath)}"
    hdfs_cmd = '-rm '
    if recursive:
        hdfs_cmd += '-r '
    if force:
        hdfs_cmd += '-f '
    cmd = '{} {}'.format(hdfs_cmd, filepath)
    try:
        check_call_hdfs_command(cmd, hadoop_binary=hadoop_binary)
        return True
    except Exception as e:
        logger.error(
            f'HDFS put command exception caught: '
            f'type {type(e).__name__}, msg: {e}')
        return False


def download_from_hdfs(src_path: str,
                       dst_path: str,
                       overwrite: bool = False,
                       raise_exception: bool = False):
    """ Download src_path from hdfs to local dst_path

    Args:
        src_path: the source hdfs path
        dst_path: the local download destination
        overwrite: if True, the local file will be overwritten if it exists
        raise_exception: if True, error is raised when thing goes wrong
    """
    # Legality check
    assert isinstance(src_path, str) and has_hdfs_path_prefix(
        src_path), src_path
    assert isinstance(dst_path, str) and not has_hdfs_path_prefix(
        dst_path), dst_path

    # Get the targeted download path
    if os.path.isdir(dst_path):  # download to an existing folder
        download_path = os.path.join(dst_path, os.path.basename(src_path))
    else:  # download as a file
        download_path = dst_path

    if overwrite is True:  # Remove the targeted file/folder if it exists
        if os.path.isdir(download_path):
            shutil.rmtree(download_path)
        elif os.path.isfile(download_path):
            os.remove(download_path)
    else:  # skip downloading if the targeted file/folder exists
        if os.path.exists(download_path):
            return True

    # Download from hdfs
    try:
        cmd = '-get {} {}'.format(src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = f'Failed to download src {src_path} to dst {dst_path}: {e}'
        if raise_exception:
            raise ValueError(msg)
        return False
