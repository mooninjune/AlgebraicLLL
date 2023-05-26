sage --preparse keflll.sage
sage --preparse util_l2.sage
echo "2 of 10 done..."
sage --preparse utils.sage
sage --preparse l2_.sage
echo "4 of 10 done..."
sage --preparse keflll_wrapper.sage
sage --preparse utils_wrapper.sage
echo "6 of 10 done..."
sage  --preparse gen_lat.sage
sage --preparse common_params.sage
echo "8 of 10 done..."
sage --preparse svp_tools.sage
sage --preparse LLL_params.sage
echo "parsing done..."

mv keflll.sage.py  keflll.py
mv util_l2.sage.py util_l2.py
mv utils.sage.py  utils.py
mv l2_.sage.py  l2_.py
mv keflll_wrapper.sage.py  keflll_wrapper.py
mv utils_wrapper.sage.py  utils_wrapper.py
mv gen_lat.sage.py  gen_lat.py
mv common_params.sage.py common_params.py
mv svp_tools.sage.py svp_tools.py
mv LLL_params.sage.py LLL_params.py

sage --preparse arakelov.sage
mv  arakelov.sage.py  arakelov.py
