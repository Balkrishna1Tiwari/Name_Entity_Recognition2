import os
from N_E_R.logging import logger
from N_E_R.entity import DataValidationConfig

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    
    def validate_all_files_exist(self)-> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts","data_ingestion","conll2003"))
            print(all_files)
            for file in all_files:
                if file!='conll2003_dataset.zip':
                    if file not in self.config.ALL_REQUIRED_FILES:
                        validation_status = False
                        with open(self.config.STATUS_FILE, 'w') as f:
                            f.write(f"Validation status: {validation_status}")
                    else:
                        validation_status = True
                        with open(self.config.STATUS_FILE, 'w') as f:
                            f.write(f"Validation status: {validation_status}")
                else:
                    continue
            return validation_status
        
        except Exception as e:
            raise e
