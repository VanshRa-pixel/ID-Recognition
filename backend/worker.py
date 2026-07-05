from concurrent.futures import ThreadPoolExecutor
from extractor import extract_information

executor = ThreadPoolExecutor(
    max_workers=10
)

def process_document(image_path):
    future = executor.submit(
        extract_information,
        image_path
    )

    return future.result()