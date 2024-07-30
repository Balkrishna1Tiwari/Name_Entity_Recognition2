from N_E_R.config.configuration import ConfigurationManager
from N_E_R.components.data_transformation import DataTransformation
from N_E_R.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation()
        
# d=DataTransformationTrainingPipeline()
# d.main()
