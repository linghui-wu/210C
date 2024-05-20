clear all

**************************
******* Part 1 VAR *******
**************************

* Construct the dataset
import delimited "data/FEDFUNDS.csv", clear
tempfile fedfunds_df
save `fedfunds_df', replace

import delimited "data/GDPDEF.csv", clear
tempfile gdpdef_df
save `gdpdef_df', replace

import delimited "data/UNRATE.csv", clear
merge 1:1 date using `fedfunds_df', assert(master match) keep(match) nogen
merge 1:1 date using `gdpdef_df', assert(master using match) keep(match) nogen

gen date_var = date(date, "YMD")
format date_var %tdCY-N

label var fedfunds "Federal Funds Effective Rate"
label var gdpdef "GDP Deflator"
label var unrate "Unemployment Rate"
label var date_var "Date"

* Set time series
tsset date_var

* Plot time series
tsline fedfunds unrate, yaxis(1) || tsline gdpdef, yaxis(2) ///
    graphregion(fcolor(white)) ///
    xlabel(, labsize(small)) ///
    ylabel(, axis(1) labsize(small)) ///
    ylabel(, axis(2) labsize(small))
graph export "figs/time_series.png", replace

* Generate quarter variable
gen quarter = qofd(date_var)

* Average across months within each quarter 
collapse (mean) fedfunds unrate gdpdef (min) date_var, by (quarter)
label var quarter  "Quarter"
format quarter %tq
tsset quarter, quarterly

* Restrict time period to 1959Q1:2007Q4
drop if quarter < -4 | quarter > 191

* Generate inflation rate from GDP deflator
gen infl = 100 * (gdpdef - gdpdef[_n-4]) / gdpdef[_n-4]
drop if quarter < 0

* Estimate reduced-form VAR
var infl unrate fedfunds

* Estimate structural VAR
matrix A = (.,0,0 \ .,.,0 \ .,.,.)
matrix B = (1,0,0 \ 0,1,0 \ 0,0,1)
svar infl unrate fedfunds, lags(1/4) aeq(A) beq(B)

* Create and plot impulse response functions (IRF)
irf create sirfs, set(sirfs) step(20) replace
irf graph sirf, irf(sirfs) impulse(infl unrate fedfunds) ///
    response(infl unrate fedfunds)  ///
    byopts(graphregion(fcolor(white)) yrescale /// 
    xrescale note("") legend(pos(3) )) legend(stack col(1) /// 
    order(1 "95% CI" 2 "SIRF") symx(*.5) size(vsmall))  ///
    xtitle("Quarters after shock")
graph export "figs/svar_irf.png", replace

* Plot time series of identified monetary shocks
predict resid_monetary, residuals equation(fedfunds)
label var resid_monetary "Monetary Shocks"
    
* Plot residuals and save graph
tsline resid_monetary, graphregion(fcolor(white)) 
graph export "figs/estimated_monetary_shocks.png", replace


***********************************
******* Part 2 Romer Shocks *******
***********************************

* Merge with Romer-Romer shocks data
rename quarter date
merge 1:1 date using "data/RR_monetary_shock_quarterly.dta", assert (master match) nogen

* Set Romer shocks to zero before 1969Q1
replace resid       = 0 if yofd(date_var) < 1969
replace resid_romer = 0 if yofd(date_var) < 1969
replace resid_full  = 0 if yofd(date_var) < 1969

* Estimate reduced-form VAR with Romer shocks as exogeneous variables
tsset date, quarterly
var infl unrate fedfunds, lags(1/8)  exog(L(0/12).resid_full)

* Construct and plot IRF
irf create rirf, step(20) replace
irf graph dm, irf(rirf) impulse(resid_full) graphregion(fcolor(white))
graph export "figs/var_rirf.png", replace

* Estimate structural VAR 
matrix AA = (.,0,0,0 \ .,.,0,0 \ .,.,.,0 \ .,.,.,.)
matrix BB = (1,0,0,0 \ 0,1,0,0 \ 0,0,1,0 \ 0,0,0,1)
svar resid_full infl unrate fedfunds, lags(1/4) aeq(AA) beq(BB)

* Create and plot impulse response functions (IRF)
irf create sirfs_rr, set(sirfs_rr) step(20) replace
irf graph sirf, irf(sirfs_rr) impulse(resid_full infl unrate fedfunds) ///
    response(resid_full infl unrate fedfunds)  ///
    byopts(graphregion(fcolor(white)) yrescale /// 
    xrescale note("") legend(pos(3) )) legend(stack col(1) /// 
    order(1 "95% CI" 2 "SIRF") symx(*.5) size(vsmall))  ///
    xtitle("Quarters after shock")
graph export "figs/svar_irf_rr.png", replace

* Plot time series of identified Romer monetary shocks
predict resid_rr_monetary, residuals equation(fedfunds)
label var resid_rr_monetary "Romer Monetary Shocks"
tsline resid_rr_monetary || tsline resid_monetary, graphregion(fcolor(white)) 
graph export "figs/compare_monetary_shocks.png", replace

