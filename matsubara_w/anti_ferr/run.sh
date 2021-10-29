#!/bin/bash
rm figures/*
rm figures/Gwn/*
rm figures/Gw0/*
rm figures/Gtau/*
rm figures/Gtau_not_converged/*
rm figures/Sigmatau_not_converged/*
rm data/*
python "ipt_imag_HM.py"
