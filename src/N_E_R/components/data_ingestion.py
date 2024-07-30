import os
import urllib.request as request
import zipfile
from pathlib import Path
from N_E_R.logging import logger
from N_E_R.utils.common import get_size
from N_E_R.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            try:
                response = request.urlopen(self.config.source_URL)
                content_type = response.info().get('Content-Type')
                if 'zip' in content_type:
                    with open(self.config.local_data_file, 'wb') as out_file:
                        out_file.write(response.read())
                    logger.info(f"{self.config.local_data_file} downloaded successfully.")
                else:
                    logger.error(f"Failed to download the file. Expected a ZIP file but got {content_type}.")
            except Exception as e:
                logger.error(f"Failed to download the file. Error: {e}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory.
        Function returns None.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
                logger.info(f"Extraction completed successfully. Files extracted to: {unzip_path}")
        except zipfile.BadZipFile:
            logger.error("Error: The file is not a valid ZIP file.")
        except FileNotFoundError:
            logger.error(f"Error: The file {self.config.local_data_file} does not exist.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

# # Example usage
# if __name__ == "__main__":
#     config = DataIngestionConfig(
#         local_data_file="dataset.zip",  # Local path to save the ZIP file
#         unzip_dir="path_to_unzip_directory",  # Directory to extract the contents of the ZIP file
#         source_URL=

#     data_ingestion = DataIngestion(config)
#     data_ingestion.download_file()
#     data_ingestion.extract_zip_file()
