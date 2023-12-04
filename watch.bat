@echo off
set AOC_COPY_CLIPBOARD=1
set MIMALLOC_LARGE_OS_PAGES=1
set MIMALLOC_PAGE_RESET=0
set RUST_BACKTRACE=1

:watch
cls
for %%i in (.\src\bin\*.rs) do (
    echo [Running 'python run.py %%~ni']
    cargo run --bin %%~ni
)

goto watch
