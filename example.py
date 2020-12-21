#!/usr/bin/env python3
from maximize_stacked_revenues import battery_portfolio

ns = 3
p_max = [10, 10, 10]
E_max = [10, 10, 10]
E0 = [0.5, 0.5, 0.5]
roundtrip_eff = [0.9, 0.9, 0.9]


dam_participation = 1
rm_participation = 1
fm_participation = 1
bm_participation = 1

# # Market Participation
# dam_participation = 1  # day ahead market participation - 0 or 1
# rm_participation = 1  # reserve market participation - 0 or 1
# fm_participation = 1  # flexibility market participation - 0 or 1
# bm_participation = 1  # balancing market participation - 0 or 1

# # Number of Batteries
# ns = 2

# # Batteries' power capacity // List - Number of items = ns
# p_max = [500, 500]

# # Batteries' energy capacity // List - Number of items = ns
# E_max = [1000, 1000]

# # Batteries' initial State-of-Energy // List - Number of items = ns
# E0 = [0.5, 0.5]

# # Batteries' roundtrip efficiencies // List - Number of items = ns
# roundtrip_eff = [0.8464, 0.8464]

# # Day Ahead energy market prices // List - Number of items = Number of timeslots
# dam_prices = [35.65, 35.54, 35.22, 35.34, 35.39, 36.03, 36.56, 39.07, 41.03, 41.74, 42.09,
#                41.67, 41.30, 40.77, 41.14, 41.98, 46.07, 46.56, 45.65, 44, 42.87, 41.92, 41.32, 39.17]
dam_prices = [1,1]

# # Reserve Market prices, Up Regulation // List - Number of items = Number of timeslots
# rup_prices = [8.80, 8.40, 8.60, 8.60, 8.60, 8.60, 8.70, 9.20, 9.05, 8.80, 8.80,
#                8.80, 8.80, 8.80, 8.80, 8.80, 9.05, 8.70, 8.70, 8.70, 8.70, 8.70, 8.70, 8.30]
rup_prices = [1,1]

# # Reserve Market prices, Down Regulation // List - Number of items = Number of timeslots
# rdn_prices = [8.80, 8.40, 8.60, 8.60, 8.60, 8.60, 8.70, 9.20, 9.05, 8.80, 8.80,
#                8.80, 8.80, 8.80, 8.80, 8.80, 9.05, 8.70, 8.70, 8.70, 8.70, 8.70, 8.70, 8.30]
rdn_prices = [1,1]

# # Flexibility Market active power prices // Multi-dimensional list (ns dimensions) - Number of items per dimension = Number of timeslots
# fmp_prices = [[-10.2730709943157, -10.2730709943157, -10.2730709943198, -10.2730709943156, -10.2730709943156, -10.2730709942963, -10.2730709943157, -9.67957014321651, -9.58557186164547, -9.58557186169831, -9.26398182001896, -15, -14.9999999999998, -15, -9.94004228256849, -9.96891422015399, -9.53790629365877, -9.53790629374786, 1.83066970565394, 1.52453365166119, 1.29608123810499, 0.262007776958364, 0.604921057426301, -
#                 10.0266431790510], [2.98175044968769, 2.98175044963742, 2.98175044933211, 2.98175044933545, 2.98175044933446, 2.98175044937664, 2.98175044950847, 11.9914519415397, 13.7400095913706, 13.7400097961880, 15.0000000000001, 3.53811224211740, 3.01012829125596, 3.01012858020784, 3.43502952365796, 3.41345222557084, 15.0000000000002, 15, 15, 12.5502493683961, 15, 12.0399595334352, 14.9999999999991, 11.0242975315517]]
fmp_prices = [[1,1],[1,1],[1,1]]

# # Flexibility Market reactive power prices // Multi-dimensional list (ns dimensions) - Number of items per dimension = Number of timeslots
# fmq_prices = [[-6.98754125928292, -6.97466268387100, -6.97466268387156, -6.97466268387182, -6.97466268387156, -6.97466268387056, -6.97466268387156, -6.89100192883195, -6.87771666278788, -6.87771666278799, -6.83238558751089, -10.5027351204000, -10.3954889132579, -10.3954889835041, -6.92768253234579, -6.93178829665201, -6.87099776264284, -6.87099776264326, 0.932572594516004, 0.776627620260265, 0.660254510076594, 1.96619942549603e-11,
#                0.308145896532837, -6.93992585510404], [0.863462774083544, 3.00000000000006, 3.00000000000000, 3.00000000000002, 3.00000000000000, 2.99999999999898, 3.00000000000001, 9.11827574663382, 10.3424142652509, 10.3424142652358, 10.9440676608277, 3.00000000000024, 3, 2.99999999999809, 3, 3.00000000012164, 11.2355695927228, 11.2355695927553, 10.5428497797808, 8.82602739559635, 10.7494763817741, 8.68620530078705, 11.0031579140256, 8.76279016906267]]
fmq_prices = [[1,1],[1,1],[1,1]]

# # Balancing Market prices - Upward // List - Number of items = Number of timeslots
# bm_up_prices = [38.33, 41.54, 41.54, 41.54, 39.56, 35.39, 36.03, 36.56, 40.93, 46.00, 47.47,
#                 47.47, 47.47, 45.99, 40.77, 41.14, 41.98, 58.00, 64.78, 65.00, 52.00, 49.45, 50.00, 47.50]
bm_up_prices = [1,1]

# # Balancing Market prices - Downward // List - Number of items = Number of timeslots
# bm_dn_prices = [38.33, 35.65, 35.54, 35.22, 35.34, 35.39, 35.50, 35.50, 39.07, 41.03, 41.74,
#                 42.09, 41.67, 41.30, 40.77, 37.09, 38.20, 39.56, 46.56, 45.65, 44.00, 42.87, 41.92, 41.32]
############################################################################################################################################
bm_dn_prices = [1,1]

# Create a battery object
bsu = battery_portfolio(ns, dam_participation, rm_participation, fm_participation, bm_participation, dam_prices, rup_prices, rdn_prices, fmp_prices, fmq_prices, bm_up_prices, bm_dn_prices,
                        p_max, E_max, roundtrip_eff, E0)

# Maximize stacked revenues
[Profits, pup, pdn, dam_schedule, rup_commitment, rdn_commitment, pflexibility, qflexibility,
    soc, DAM_profits, RM_profits, FM_profits, BM_profits] = bsu.maximize_stacked_revenues()

# Print total profits
print([Profits, pup, pdn, dam_schedule, rup_commitment, rdn_commitment,
       pflexibility, qflexibility, soc, DAM_profits, RM_profits, FM_profits, BM_profits])
