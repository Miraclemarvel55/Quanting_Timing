#!/usr/bin/env python
# -*- coding: utf-8 -*-
import data.update_data as updt_dt
import data.data_const as dtcst
import ml_stock.ml_all as ml
import make_envs_dir_satisfied as mk_ok

def sap_main():
    mk_ok.mk_all_dir_ok()
    realtime_data = updt_dt.update_all()
    ml.ml_all(realtime_data)