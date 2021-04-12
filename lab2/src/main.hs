import           Data.Char
import           Data.Complex
import           Debug.Trace
import           System.Environment
list1 :: [Double]
list1 = [1.0000000000000000,
 0.5772156649015329,
 -0.6558780715202538,
 -0.0420026350340952,
 0.1665386113822915,
 -0.0421977345555443,
 -0.0096219715278770,
 0.0072189432466630,
 -0.0011651675918591,
 -0.0002152416741149,
 0.0001280502823882,
 -0.0000201348547807,
 -0.0000012504934821,
 0.0000011330272320,
 -0.0000002056338417,
 0.0000000061160950,
 0.0000000050020075,
 -0.0000000011812746,
 0.0000000001043427,
 0.0000000000077823,
 -0.0000000000036968,
 0.0000000000005100,
 -0.0000000000000206,
 -0.0000000000000054,
 0.0000000000000014,
 0.0000000000000001]


mpow :: (Double, Int) -> Double
mpow(x,n) | n == 0 = 1
          | otherwise = x * mpow(x, n - 1)

mfac :: (Int) -> Integer
mfac (n)      | n < 0 = -1
              | n <= 1 = 1
	         | otherwise = fromIntegral(n) * mfac(n-1)


mygamma :: (Double, Int) -> Double
mygamma (x,n) | n > 26 = 0
	         | otherwise =  (list1 !! (n-1)) * (x^n) + mygamma(x,n+1)

mg :: (Double) -> Double
mg (x)        | x < 1 = 0
              | x > 2 = (x-1) * mg(x-1)
              | otherwise = 1/mygamma(x,1)

rec_mg :: (Double, Double) -> Double
rec_mg (n, z)
               | n == 1 = (1/mygamma(1 + z, 1))
               | otherwise = (n - 1 + z) * rec_mg(n - 1, z)

foo :: (Double) -> Double
foo (v)        | truncate(v) == 0 = v
               | otherwise = v - fromIntegral(truncate(v))

ryadfunc :: (Double, Complex Double, Int) -> Complex Double
ryadfunc (v, z, k) = (((z*z)/(4.0 :+ 0)) **(fromInteger(fromIntegral(k)) :+ 0))/((fromInteger(mfac(k)) * rec_mg(fromIntegral(truncate(v + fromInteger(fromIntegral(k)) + 1)), foo(v + fromInteger(fromIntegral(k)) + 1))) :+ 0)

bessel_i_ryad :: (Double, Complex Double, Int) -> Complex Double
bessel_i_ryad (v, z, k)       | k == -1 = 0
			               | otherwise = ryadfunc(v, z, k) + bessel_i_ryad(v, z, k - 1)

prom_func_chisl :: (Double, Int) -> Complex Double
prom_func_chisl (m, poryadok) | poryadok == 0 = 1 :+ 0
                              | otherwise = ((m - ((fromInteger(fromIntegral(poryadok + poryadok - 1)))**2)) :+ 0) * prom_func_chisl (m, poryadok - 1)

prom_func :: (Double, Complex Double, Int) -> Complex Double
prom_func (v, z, poryadok) | poryadok == 0 = 1 :+ 0
			            | otherwise = (prom_func_chisl(4 * v * v * v, poryadok)/((fromInteger(mfac(poryadok)) :+ 0) * ((((-8) :+ 0) * z)^poryadok))) + prom_func(v, z, poryadok - 1)

assimp_bessel_i :: (Double, Complex Double, Int) -> Complex Double
assimp_bessel_i (v, z, poryadok) = ((exp(z))/sqrt(((2*3.14159265) :+ 0)*z)) * prom_func(v, z, poryadok)

rec_bess_I :: (Double, Complex Double, Int, Int, Complex Double, Complex Double, Double) -> Complex Double
rec_bess_I (v, z, k, poryadok, prev_I, now_I, now_v)    | v == now_v = now_I
                                                        | otherwise = rec_bess_I(v, z, k, poryadok, now_I, prev_I - (now_I*(2 :+ 0)*(now_v :+ 0)/z), now_v+1)
rec_bess_I_minus :: (Double, Complex Double, Int, Int, Complex Double, Complex Double, Double) -> Complex Double
rec_bess_I_minus (v, z, k, poryadok, prev_I, now_I, now_v)     | (((v * now_v) > 0) && (truncate(v) == truncate(now_v))) = now_I
                                                               | otherwise = rec_bess_I_minus(v, z, k, poryadok, now_I, ((2*now_v :+ 0)*now_I/z) + prev_I, now_v - 1)
kamanda :: (Double) -> Double
kamanda (v)         | ((-v + fromInteger(truncate v)) == 0) = 0
                    | otherwise = 1 - (-v + fromInteger(truncate(v)))

bessel_i :: (Double, Complex Double, Int, Int) -> Complex Double
bessel_i (v, z, k, poryadok)       | (v < 0) = rec_bess_I_minus(v, z, k, poryadok, bessel_i(1 + kamanda(v), z, k, poryadok), bessel_i(0 + kamanda(v), z, k, poryadok), 0 + kamanda(v))
                                   | (v >= 2) = rec_bess_I (v, z, k, poryadok, bessel_i(0 + (v - fromInteger(truncate(v))), z, k, poryadok), bessel_i(1 + (v - fromInteger(truncate(v))), z, k, poryadok), 1 + (v - fromInteger(truncate(v))))
                                   | (magnitude(z) >= 99) = assimp_bessel_i(v, z, poryadok) --зачем ?
                                   | otherwise = ((z/(2.0 :+ 0))**(v :+ 0)) * bessel_i_ryad(v, z, k)
psi_sum :: (Int) -> Double
psi_sum (n)    | (n - 1) == 0 = 0
               | otherwise = fromInteger(fromIntegral(((n-1))))**(-1) + psi_sum(n - 1)

psi :: (Int) -> Double
psi (n)   | n == 1 = -0.5772156649
          | n >= 2 = -0.5772156649 + psi_sum(n)

first_k_sum :: (Int, Int, Complex Double) -> Complex Double
first_k_sum (n, k, z)  | k == n = 0
                       | otherwise = (fromInteger(mfac(n - k - 1))/fromInteger(mfac(k))) * ((-(z**2)/4)^^k) + first_k_sum(n, k+1, z)
last_k_sum :: (Int, Complex Double, Int, Int) -> Complex Double
last_k_sum (n, z, k, poryadok)     | k == poryadok = 0 :+ 0
                                   | otherwise = ((psi(k + 1) + psi(n + k + 1)) :+ 0) * ((((z**(2 :+ 0))/4)^k)/(fromInteger(mfac(k)*mfac(n+k)))) + last_k_sum(n, z, k+1, poryadok)
prom_func_k :: (Double, Complex Double, Int) -> Complex Double
prom_func_k (v, z, poryadok) | poryadok == 0 = 1 :+ 0
			              | otherwise = (prom_func_chisl(4 * v * v * v, poryadok)/(fromInteger(mfac(poryadok)) * ((8 * z)^^poryadok))) + prom_func(v, z, poryadok - 1)

assimp_bessel_k :: (Double, Complex Double, Int) -> Complex Double
assimp_bessel_k (v, z, poryadok) = ((2.718281828**(-z))*sqrt(3.14159265/(z*2))) * prom_func_k(v, z, poryadok)

rec_bess_k :: (Double, Complex Double, Int, Int, Complex Double, Complex Double, Double) -> Complex Double
rec_bess_k (v, z, k, poryadok, prev_I, now_I, now_v)    | v == now_v = now_I
                                                        | otherwise = rec_bess_k(v, z, k, poryadok, now_I, ((prev_I * (mkPolar 1 ((now_v - 1) * 3.14159265))) - (now_I*(((2*now_v) :+ 0)/z) * (mkPolar 1 (now_v * 3.14159265)))) / (mkPolar 1 ((now_v + 1) * 3.14159265)) , now_v+1)

rec_bess_k_minus :: (Double, Complex Double, Int, Int, Complex Double, Complex Double, Double) -> Complex Double
rec_bess_k_minus (v, z, k, poryadok, prev_I, now_I, now_v)     | (((v * now_v) > 0) && (truncate(v) == truncate(now_v))) = now_I
                                                               | otherwise = rec_bess_k_minus(v, z, k, poryadok, now_I, ((now_I*(((2*now_v) :+ 0)/z) * (mkPolar 1 (now_v * 3.14159265))) + (prev_I * (mkPolar 1 ((now_v + 1) * 3.14159265)))) / (mkPolar 1 ((now_v - 1) * 3.14159265)), now_v - 1)
ryad_bessel_k :: (Int, Complex Double, Int, Int) -> Complex Double
ryad_bessel_k (n, z, k, poryadok)   | magnitude(z) >= 9 = assimp_bessel_k(fromInteger(fromIntegral(n)), z, poryadok)
                                    | otherwise = (1/2) * ((z/2)^^(-n)) * first_k_sum(n, 0, z) + ((-1)^^(n+1)) * log(z/2) * bessel_i(fromInteger(fromIntegral(n)), z, k, poryadok) + ((-1)^^n) * 0.5 * ((z/2)^^n) * last_k_sum(n, z, 0, k)

func_bessel_k_from_i :: (Double, Complex Double, Int, Int) -> Complex Double
func_bessel_k_from_i (n, z, k, poryadok) | magnitude(z) > 15 = assimp_bessel_k(n, z, poryadok)
                                         | otherwise = (3.14159265/2) * (bessel_i(-n, z, k, poryadok) - bessel_i (n, z, k, poryadok))/(sin(n*3.14159265) :+ 0)

bessel_k :: (Double, Complex Double, Int, Int) -> Complex Double
bessel_k (n, z, k, poryadok)
                              | n >= 2 = rec_bess_k(n, z, k, poryadok, bessel_k(0 + (n - fromInteger(truncate(n))), z, k, poryadok), bessel_k(1 + (n - fromInteger(truncate(n))), z, k, poryadok), 1 + (n - fromInteger(truncate(n))))
                              | n < 0 = rec_bess_k_minus(n, z, k, poryadok, bessel_k(1 + kamanda(n), z, k, poryadok), bessel_k(0 + kamanda(n), z, k, poryadok), 0 + kamanda(n))
                              | ((n - fromInteger(truncate(n))) == 0) = ryad_bessel_k(truncate(n), z, k, poryadok)
                              | otherwise = func_bessel_k_from_i (n, z, k, poryadok)

main = do
     args <- getArgs
     let label = args!!6
     if (args!!5 == "I") then do
              putStrLn (show(realPart(bessel_i(read(args!!0),read(args!!1) :+ read(args!!2),read(args!!3),read(args!!4)))))
              appendFile label (args!!1 ++ "\t" ++ show(realPart(bessel_i(read(args!!0),read(args!!1) :+ read(args!!2),read(args!!3),read(args!!4)))) ++ "\t" ++ show(imagPart(bessel_i(read(args!!0),read(args!!1) :+ read(args!!2),read(args!!3),read(args!!4)))) ++ "\n")
     else do
            putStrLn (show(realPart(bessel_k(read(args!!0),read(args!!1) :+ read(args!!2),read(args!!3),read(args!!4)))))
            appendFile label (args!!1 ++ "\t" ++ show(realPart(bessel_k(read(args!!0),read(args!!1) :+ read(args!!2),read(args!!3),read(args!!4)))) ++ "\t" ++ show(imagPart(bessel_k(read(args!!0),read(args!!1) :+ read(args!!2),read(args!!3),read(args!!4)))) ++ "\n")
