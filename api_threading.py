import asyncio
import json
from typing import List
import logging
import os

try:
    # Try a relative import (when run as part of a package)
    from .parallel_processor import (
        process_api_requests_from_file,
        api_endpoint_from_url,
        num_tokens_consumed_from_request,
        APIRequest,
        StatusTracker
    )
except ImportError:
    # Fall back to an absolute import (when run as a standalone script)
    from parallel_processor import (   process_api_requests_from_file,     api_endpoint_from_url,  num_tokens_consumed_from_request,    APIRequest,     StatusTracker    )


async def execute_api_requests_in_parallel(
        request_strings: List[str],
        save_filepath: str,
        request_url: str,
        api_key: str,
        max_requests_per_minute: float =3_000 * 0.2,
        max_tokens_per_minute: float =250_000 * 0.2,
        token_encoding_name: str = "cl100k_base",
        max_attempts: int = 3,
        logging_level: int = logging.INFO,
):
    # create list of request JSON objects
    requests = [json.loads(request_string) for request_string in request_strings]

    # create temp file
    abs_filepath = os.path.abspath(save_filepath)  # Convert to absolute path
    dir_path = os.path.dirname(abs_filepath)  # Extract directory name
    base_filename = os.path.splitext(os.path.basename(abs_filepath))[0]

    # Prepare the file path with the required name ending
    file_path = os.path.join(dir_path, base_filename + "_log.txt")

    # Ensure that the directory exists
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w") as f:
        for job in requests:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}


    # run tasks in parallel

    await process_api_requests_from_file(
        requests_filepath=file_path,
        save_filepath=save_filepath,
        request_url=request_url,
        api_key=api_key,
        max_requests_per_minute=float(max_requests_per_minute),
        max_tokens_per_minute=float(max_tokens_per_minute),
        token_encoding_name=token_encoding_name,
        max_attempts=int(max_attempts),
        logging_level=int(logging_level),
    )
