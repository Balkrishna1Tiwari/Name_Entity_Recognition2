from N_E_R.config.configuration import ConfigurationManager
from N_E_R.components.data_validation import DataValiadtion
from N_E_R.logging import logger


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_files_exist()
        
# d=DataValidationTrainingPipeline()
# d.main()