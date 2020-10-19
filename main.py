from scripts.helper.Widgets import Widgets
from scripts.constants import *
from scripts.data_processing.DataProcessor import DataProcessor
from scripts.indicator_calculation.IndicatorCalculator import IndicatorCalculator
from scripts.models.Models import Models
from scripts.models.Arima import Arima
from scripts.models.Garch import Garch
from scripts.models.Neural import NeuralNetworkModels

dp = DataProcessor()
wg = Widgets()
ic = IndicatorCalculator()

file_list = DataProcessor.getFileName(dp)

# #get the file name, that needs to be processed
file_name = Widgets.askToSelectOption(wg,
  		    file_selection_prompt,
  		    file_list, (15, 26))
# #get the format of file
header_format = Widgets.userEntry(wg, file_format_prompt, header, enter_button_text)

DataProcessor.imputeData(dp, file_name, header_format, header)

IndicatorCalculator.calculate_indicators(ic, file_name)
#file_name = "NIFTY-I.csv"
# #get the data and split info
data_info = Widgets.userEntry(wg, train_test_split_prompt, data_info_prompt, enter_button_text)
#data_info = ["600","60","10","10","10"]
#get the type of prediction
prediction_type = Widgets.scrollbar(wg, prediction_type_prompt, prediction_type_option)

if(prediction_type_option.index(prediction_type) < 3):
	filename_updated = file_name[:-4] + '_yoc.csv'
	predict_col      = oc_dictionary[prediction_type]
else:
	filename_updated = file_name[:-4] + '_ycc.csv'
	predict_col      = cc_dictionary[prediction_type]

# -1 in place of features to be updated for auto and manual
#mod = Models(filename_updated, data_info, -1, prediction_type, predict_col)
#Models.RunAll(mod)
arima_model = Arima(filename_updated, data_info, prediction_type,predict_col)
#arima_model.DickeyFuller()
#arima_model.ReturnAllPredicted()
arima_model.PlotData()

#garch_model = Garch(filename_updated, data_info, prediction_type,predict_col)
#garch_model.ReturnAllPredicted()
# mod = NeuralNetworkModels(filename_updated, data_info, -1, prediction_type, predict_col)
# NeuralNetworkModels.RunAll(mod)
