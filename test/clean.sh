rm *.so
rm -r config_cache
find ../src/ \( -name "*.so" -o -name "*.o" \) -exec gio trash {} +
find ../src/ \( -name "*.c" -o -name "*.c"  \) -exec gio trash {} +