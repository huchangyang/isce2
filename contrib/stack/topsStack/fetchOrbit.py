#!/usr/bin/env python3

from types import SimpleNamespace
import json
import requests
import re
import os
import argparse
import datetime
import time
import glob

orbitMap = [('precise', 'AUX_POEORB'),
            ('restituted', 'AUX_RESORB')]

datefmt = "%Y%m%dT%H%M%S"
queryfmt = "%Y-%m-%d"

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Fetch orbits corresponding to given SAFE package or directory')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='Path to SAFE package or directory containing SAFE files')
    parser.add_argument('-o', '--output', dest='outdir', type=str, default='.',
                        help='Path to output directory')
    parser.add_argument('-t', '--token-file', dest='token_file', type=str, default='.copernicus_dataspace_token',
                        help='Filename to save auth token file')
    parser.add_argument('-u', '--username', dest='username', type=str, default=None,
                        help='Copernicus Data Space Ecosystem username')
    parser.add_argument('-p', '--password', dest='password', type=str, default=None,
                        help='Copernicus Data Space Ecosystem password')

    return parser.parse_args()


def FileToTimeStamp(safename):
    '''
    Return timestamp from SAFE name.
    '''
    safename = os.path.basename(safename)
    fields = safename.split('_')
    sstamp = []  # sstamp for getting SAFE file start time, not needed for orbit file timestamps

    try:
        tstamp = datetime.datetime.strptime(fields[-4], datefmt)
        sstamp = datetime.datetime.strptime(fields[-5], datefmt)
    except:
        p = re.compile(r'(?<=_)\d{8}')
        match = p.search(safename)
        if match is None:
            raise ValueError(f"Cannot parse timestamp from filename: {safename}")
        dt2 = match.group()
        tstamp = datetime.datetime.strptime(dt2, '%Y%m%d')

    satName = fields[0]

    return tstamp, satName, sstamp


def find_safe_files(input_path):
    '''
    Find all SAFE files in the given path (file or directory).
    '''
    safe_files = []
    
    if os.path.isfile(input_path):
        # Single file input
        if input_path.endswith(('.zip', '.SAFE')):
            safe_files.append(input_path)
        else:
            print(f"Warning: {input_path} is not a SAFE file (should end with .zip or .SAFE)")
    elif os.path.isdir(input_path):
        # Directory input - find all SAFE files
        patterns = ['*.zip', '*.SAFE', '**/*.zip', '**/*.SAFE']
        for pattern in patterns:
            safe_files.extend(glob.glob(os.path.join(input_path, pattern), recursive=True))
        
        if not safe_files:
            print(f"No SAFE files found in directory: {input_path}")
        else:
            print(f"Found {len(safe_files)} SAFE files in directory: {input_path}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    return safe_files


def get_saved_token_data(token_file):
    try:
        with open(token_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None


def save_token_data(access_token, expires_in, token_file):
    token_data = {
        "access_token": access_token,
        "expires_at": time.time() + expires_in
    }
    with open(token_file, 'w') as file:
        json.dump(token_data, file)


def is_token_valid(token_data):
    if token_data and "expires_at" in token_data:
        return time.time() < token_data["expires_at"]
    return False


def get_new_token(username, password, session):
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    response = session.post(url, data=data)
    response.raise_for_status()
    token_info = response.json()
    return token_info["access_token"], token_info["expires_in"]


def download_file(file_id, outdir='.', session=None, token=None, max_retries=3):
    '''
    Download file to specified directory with retry mechanism.
    '''
    if session is None:
        session = requests.session()

    url = "https://zipper.dataspace.copernicus.eu/odata/v1/"
    url += f"Products({file_id})/$value"

    path = outdir
    print('Downloading URL: ', url)
    
    for attempt in range(max_retries):
        try:
            request = session.get(url, stream=True, verify=True,
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=30)  # 添加超时设置
            
            request.raise_for_status()
            
            with open(path, 'wb') as f:
                for chunk in request.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            
            print(f"Successfully downloaded to: {path}")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"Authentication failed (401 Unauthorized) - token may be expired")
                return False  # 返回 False 表示需要重新获取令牌
            else:
                print(f"HTTP error {e.response.status_code}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"All {max_retries} attempts failed")
                    return False
        except requests.exceptions.RequestException as e:
            print(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"All {max_retries} attempts failed")
                return False
        except Exception as e:
            print(f"Unexpected error during download: {str(e)}")
            return False
    
    return False


def process_single_safe_file(safe_file, outdir, session, token, username, password, token_file):
    '''
    Process a single SAFE file to download its orbit.
    '''
    try:
        fileTS, satName, fileTSStart = FileToTimeStamp(safe_file)
        print(f'Processing: {os.path.basename(safe_file)}')
        print('Reference time: ', fileTS)
        print('Satellite name: ', satName)
        
        match = None
        
        for spec in orbitMap:
            delta = datetime.timedelta(days=1)
            timebef = (fileTS - delta).strftime(queryfmt)
            timeaft = (fileTS + delta).strftime(queryfmt)
            url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

            start_time = timebef + "T00:00:00.000Z"
            stop_time = timeaft + "T23:59:59.999Z"

            query = " and ".join([
                f"ContentDate/Start gt '{start_time}'",
                f"ContentDate/Start lt '{stop_time}'",
                f"ContentDate/End gt '{start_time}'",
                f"ContentDate/End lt '{stop_time}'",
                f"startswith(Name,'{satName}')",
                f"contains(Name,'{spec[1]}')",
            ])

            success = False
            match = None

            try:
                r = session.get(url, verify=True, params={"$filter": query})
                r.raise_for_status()

                entries = json.loads(r.text, object_hook=lambda x: SimpleNamespace(**x)).value

                for entry in entries:
                    entry_datefmt = "%Y-%m-%dT%H:%M:%S.000000Z"
                    tbef = datetime.datetime.strptime(entry.ContentDate.Start, entry_datefmt)
                    taft = datetime.datetime.strptime(entry.ContentDate.End, entry_datefmt)
                    if (tbef <= fileTSStart) and (taft >= fileTS):
                        matchFileName = entry.Name
                        match = entry.Id

                if match is not None:
                    success = True
            except Exception as e:
                print(f"Error querying orbit database: {str(e)}")
                continue

            if success:
                break

        if match is not None:
            # 为每个文件创建唯一的输出路径
            safe_basename = os.path.splitext(os.path.basename(safe_file))[0]
            output_filename = f"{safe_basename}_{matchFileName}"
            output = os.path.join(outdir, output_filename)
            
            res = download_file(match, output, session, token)
            if res is False:
                print('Failed to download orbit ID:', match)
                return False, True  # 返回 (success, need_new_token)
            else:
                print(f'Successfully downloaded orbit: {output_filename}')
                return True, False  # 返回 (success, need_new_token)
        else:
            print('Failed to find orbits for tref {0}'.format(fileTS))
            return False, False  # 返回 (success, need_new_token)
            
    except Exception as e:
        print(f'Error processing {safe_file}: {str(e)}')
        return False, False  # 返回 (success, need_new_token)


def get_fresh_token(username, password, session, token_file):
    '''
    Get a fresh authentication token.
    '''
    print("Getting fresh authentication token...")
    try:
        if username is None or password is None:
            try:
                import netrc
                host = "dataspace.copernicus.eu"
                creds = netrc.netrc().hosts[host]
            except:
                if username is None:
                    username = input("Username: ")
                if password is None:
                    from getpass import getpass
                    password = getpass("Password (will not be displayed): ")
            else:
                if username is None:
                    username, _, _ = creds
                if password is None:
                    _, _, password = creds
        
        token, expires_in = get_new_token(username, password, session)
        save_token_data(token, expires_in, token_file)
        print("New token obtained successfully")
        return token
    except Exception as e:
        print(f"Failed to get new token: {str(e)}")
        return None


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    username = inps.username
    password = inps.password
    token_file = os.path.expanduser(inps.token_file)

    # Find all SAFE files
    safe_files = find_safe_files(inps.input)
    
    if not safe_files:
        print("No SAFE files found to process.")
        exit(1)
    
    print(f"Found {len(safe_files)} SAFE files to process")
    
    # Process each SAFE file
    session = requests.Session()
    token = None
    success_count = 0
    failed_files = []
    
    for i, safe_file in enumerate(safe_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(safe_files)}: {os.path.basename(safe_file)}")
        
        try:
            # 检查令牌是否有效
            if token is None:
                token_data = get_saved_token_data(token_file)
                if token_data and is_token_valid(token_data):
                    print("Using saved access token")
                    token = token_data["access_token"]
                else:
                    token = get_fresh_token(username, password, session, token_file)
                    if token is None:
                        print("Failed to get authentication token. Skipping remaining files.")
                        break
            
            success, need_new_token = process_single_safe_file(safe_file, inps.outdir, session, token, 
                                                             username, password, token_file)
            
            if need_new_token:
                print("Token expired, getting new token...")
                token = get_fresh_token(username, password, session, token_file)
                if token is None:
                    print("Failed to get new token. Skipping remaining files.")
                    break
                # 重试当前文件
                print("Retrying current file with new token...")
                success, _ = process_single_safe_file(safe_file, inps.outdir, session, token, 
                                                    username, password, token_file)
            
            if success:
                success_count += 1
            else:
                failed_files.append(safe_file)
                
        except KeyboardInterrupt:
            print(f"\n\nProcess interrupted by user. Processed {success_count}/{i} files so far.")
            break
        except Exception as e:
            print(f"Unexpected error processing {safe_file}: {str(e)}")
            failed_files.append(safe_file)
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(safe_files)} files processed successfully")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
