from functions import *

class method:

    # Define folder paths
    input_cannabis_folder = 'Data/Cannabis/Input' 
    output_cannabis_folder = 'Data/Cannabis/Database'

    logger.info('<<<<<Starting data transformation for Cannabis>>>>>')

    process_canabis_files(input_cannabis_folder, output_cannabis_folder)

    logger.info('<<<<<Finished data transformation for Cannabis>>>>>')