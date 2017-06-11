import data
import preprocessing as pp

data.export(pp.pre_process(data.import_full(), categorical=True), "categorical_new")
