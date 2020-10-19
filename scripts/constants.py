enter_button_text = "OK"
file_selection_prompt = "Select the file"
file_format_prompt = "Enter the Column number"
header = ["Date", "Open", "High", "Low", "Close", "Volume"]
train_test_split_prompt = "Please enter the data"
data_info_prompt = ["Training Data Start", "Training Data Size", "Test Data Size", "Window Size", "Number of Windows"] 
prediction_type_prompt = "Select type of prediction"
oc_dictionary = {"Open/Close 1day absolute" : 121,
				"Open/Close 1day signed" : 123, "Open/Close 1day 2level quantized" : 122}
cc_dictionary = {"Close/Close 1day absolute" : 120,"Close/Close 1day signed" : 121,
				"Close/Close 1day 2level quantized" : 122, "Close/Close 1day 3level quantized" : 123,
				"Close/Close 2day absolute" : 124,"Close/Close 2day signed" : 125,
				"Close/Close 2day 2level quantized" : 126, "Close/Close 2day 3level quantized" : 127,
				"Close/Close 3day absolute" : 128,"Close/Close 3day signed" : 129,
				"Close/Close 3day 2level quantized" : 130, "Close/Close 3day 3level quantized" : 131,
				"Close/Close 4day absolute" : 132,"Close/Close 4day signed" : 133,
				"Close/Close 4day 2level quantized" : 134, "Close/Close 4day 3level quantized" : 135,
				"Close/Close 5day absolute" : 136,"Close/Close 5day signed" : 137,
				"Close/Close 5day 2level quantized" : 138, "Close/Close 5day 3level quantized" : 139,
				"Close/Close 6day absolute" : 140,"Close/Close 6day signed" : 141,
				"Close/Close 6day 2level quantized" : 142, "Close/Close 6day 3level quantized" : 143,
				"Close/Close 7day absolute" : 144,"Close/Close 7day signed" : 145,
				"Close/Close 7day 2level quantized" : 146, "Close/Close 7day 3level quantized" : 147,
				"Close/Close 8day absolute" : 148,"Close/Close 8day signed" : 149,
				"Close/Close 8day 2level quantized" : 150, "Close/Close 8day 3level quantized" : 151,
				"Close/Close 9day absolute" : 152,"Close/Close 9day signed" : 153,
				"Close/Close 9day 2level quantized" : 154, "Close/Close 9day 3level quantized" : 155,
				"Close/Close 10day absolute" : 156,"Close/Close 10day signed" : 157,
				"Close/Close 10day 2level quantized" : 158, "Close/Close 10day 3level quantized" : 159,
				"Close/Close 15day absolute" : 160,"Close/Close 15day signed" : 161,
				"Close/Close 15day 2level quantized" : 162, "Close/Close 15day 3level quantized" : 163,
				"Close/Close 20day absolute" : 164,"Close/Close 20day signed" : 165,
				"Close/Close 20day 2level quantized" : 166, "Close/Close 20day 3level quantized" : 167,
				"Close/Close 30day absolute" : 168,"Close/Close 30day signed" : 169,
				"Close/Close 30day 2level quantized" : 170, "Close/Close 30day 3level quantized" : 171}
prediction_type_option = list(oc_dictionary.keys()) + list(cc_dictionary.keys())