'''
test doc for delta_hedging.py
'''
test_sample = { 'MC_array' : True }


#For Monto-carlo stimulation
#pre-parameters: T = 2, daily_step = 4, mu = 0, sigma = 0.3, S_0 = 100, K = 100, r = 0.04, type_='call', path_num = 2
#construct the (path_num * 5, T * daily_step + 1) ndarray
if test_sample['MC_array']:
	T, daily_step, mu, sigma, S_0, K, r, type_, path_num = (2, 4, 0, 0.3, 100, 100, 0.04, 'call', 2)
	steps = T * daily_step + 1
	print ('total steps : ', steps)# 2 * 4 + 1 = 9

	MC_Paths = np.empty(shape = (path_num * 5 * steps), dtype = float, order = 'C')
	print ('MA_Paths constructed, with empty arr({},{})'.format(path_num * 5, steps)) # 10, 9

	T_left = np.linspace(T / 360, 0.000001, num=steps, endpoint=True, dtype=float)
	print ('T_left series:')
	print (T_left)

	ts_year = np.linspace(0.000001, T / 360, num=steps, endpoint=True, dtype=float)
	print ('ts_year series:')
	print (ts_year)

	W = np.concatenate((np.zeros((path_num, 1), dtype=float), np.random.randn(path_num, steps-1)), axis=1)
	print ('Randn({}, {}) arr:'.format(path_num, steps))
	print (W)
	W = np.cumsum(W, axis=1) * np.sqrt(1 / daily_step / 360)
	print ('Wt series:')
	print (W)
	X = (mu - 0.5 * sigma ** 2) * ts_year + sigma * W
	BS_Spots = S_0 * np.exp(X)
	print ('BS model Spot prices:')
	print (BS_Spots)

	col_idxs = np.tile(np.array(range(steps), dtype=int, order='C'), path_num).reshape(path_num, steps)
	rows = np.array(range(path_num), dtype=int, order='C').reshape(-1, 1) * 5 * steps
	T_left_idxs = (col_idxs + rows).ravel()
	print ('The T_left indexs is:')
	print (T_left_idxs)
	S_price_idxs = (col_idxs + rows + steps).ravel()
	print ('The Spot prices indexs is:')
	print (S_price_idxs)

	MC_Paths.put(T_left_idxs, T_left)
	print ('Put the T_lef series into MC_Paths, which is MC_Paths[ (0, 5), : ]:')
	print (MC_Paths.reshape((path_num*5, steps))[ (0, 5), ])

	MC_Paths.put(S_price_idxs, BS_Spots)
	print ('Put the Spot prices into MC_Paths, which is MC_Paths[ (1, 6), : ]:')
	print (MC_Paths.reshape((path_num*5, steps))[ (1, 6), ])
	print ('the first original stimulated Spot prices series:')
	print (BS_Spots[0,])

	MC_Paths = MC_Paths.reshape(path_num*5, steps)
	for i in range(path_num):
		log_S_div_K = np.log( np.divide( MC_Paths[i * 5 + 1, ], K ) )
		sig_mul_sqrT = np.multiply( np.sqrt(MC_Paths[i * 5, ]), sigma )
		r_add_sig2_mul_T = np.multiply( ( r + 0.5 * sigma ** 2 ), MC_Paths[i * 5, ] )
		MC_Paths[i * 5 + 2, ] = np.divide( np.add( log_S_div_K, r_add_sig2_mul_T ),  sig_mul_sqrT)#path[2]: d_1
		MC_Paths[i * 5 + 3, ] = norm.cdf(MC_Paths[i * 5 + 2, ])#path[3]: delta for call
		pdf_d_1 = norm.pdf(MC_Paths[i * 5 + 2])
		sig_mul_S_mul_sqrT = np.multiply(sigma, np.multiply(MC_Paths[i * 5 + 1, ], np.power(MC_Paths[i * 5, ], 0.5) ) )
		MC_Paths[i * 5 + 4, ] = np.divide( pdf_d_1, sig_mul_S_mul_sqrT)

	#test the d_1, delta, gamma
	print ("Test for Paths[0]....")
	print ("T_left[2]:{:.6f}, S[2]: {:.2f}, d_1[2]:{:.4f}, delta:{:.4f}, gamma:{:.4f} ".format(MC_Paths[0, 2], MC_Paths[1, 2], MC_Paths[2, 2], MC_Paths[3, 2], MC_Paths[4, 2]))
	d_1 = d_j(1, MC_Paths[1, 2], K, r, sigma, MC_Paths[0, 2])
	delta_ = get_greek(MC_Paths[1, 2], K, r, sigma, MC_Paths[0, 2])
	gamma_ = get_greek(MC_Paths[1, 2], K, r, sigma, MC_Paths[0, 2], 'gamma')
	print ("The correct d_1[2]:{:.4f}, delta[2]:{:.4f}, gamma:{:.4f} ".format(d_1, delta_, gamma_))
	print ("Test for Paths[1]....")
	print ("T_left[3]:{:.6f}, S[3]: {:.2f}, d_1[3]:{:.4f}, delta:{:.4f}, gamma:{:.4f} ".format(MC_Paths[5, 2], MC_Paths[6, 2], MC_Paths[7, 2], MC_Paths[8, 2], MC_Paths[9, 2]))
	d_1 = d_j(1, MC_Paths[6, 2], K, r, sigma, MC_Paths[5, 2])
	delta_ = get_greek(MC_Paths[6, 2], K, r, sigma, MC_Paths[5, 2])
	gamma_ = get_greek(MC_Paths[6, 2], K, r, sigma, MC_Paths[5, 2], 'gamma')
	print ("The correct d_1[2]:{:.4f}, delta[2]:{:.4f}, gamma:{:.4f} ".format(d_1, delta_, gamma_))
