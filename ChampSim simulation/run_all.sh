#!/bin/bash

mkdir -p reports
mkdir -p raw_logs

BIN=~/ChampSim/bin/champsim
TRACE_DIR=/home/venu_reddy/spec2006

WARM=2000000
SIM=10000000

TRACES=(
bwaves_1609B.trace.xz
gcc_13B.trace.xz
mcf_158B.trace.xz
omnetpp_340B.trace.xz
xalancbmk_748B.trace.xz
cactusADM_1039B.trace.xz
lbm_1004B.trace.xz
mcf_46B.trace.xz
wrf_1650B.trace.xz
)

MASTER="reports/all_results.txt"
echo "ChampSim Proxy Evaluation Results" > "$MASTER"
echo "Generated: $(date)" >> "$MASTER"
echo "==================================================" >> "$MASTER"

for T in "${TRACES[@]}"
do
    TRACE="$TRACE_DIR/$T"

    [ ! -f "$TRACE" ] && echo "Skipping $T" && continue

    NAME=${T%.trace.xz}
    LOG="raw_logs/${NAME}.log"
    REP="reports/${NAME}_report.txt"

    echo "Running $NAME ..."

    $BIN --warmup-instructions $WARM \
         --simulation-instructions $SIM \
         "$TRACE" > "$LOG"

    # ----------------------------
    # Core Stats
    # ----------------------------
    IPC=$(grep "CPU 0 cumulative IPC:" "$LOG" | awk '{print $5}')
    INS=$(grep "CPU 0 cumulative IPC:" "$LOG" | awk '{print $7}')
    CYC=$(grep "CPU 0 cumulative IPC:" "$LOG" | awk '{print $9}')

    IPC=${IPC:-0}
    INS=${INS:-0}
    CYC=${CYC:-1}

    # ----------------------------
    # Prefetch Stats
    # ----------------------------
    PF=$(grep "cpu0->cpu0_L2C PREFETCH REQUESTED:" "$LOG")

    PFREQ=$(echo "$PF" | awk '{for(i=1;i<=NF;i++) if($i=="REQUESTED:") print $(i+1)}')
    PFISS=$(echo "$PF" | awk '{for(i=1;i<=NF;i++) if($i=="ISSUED:") print $(i+1)}')
    PFUSE=$(echo "$PF" | awk '{for(i=1;i<=NF;i++) if($i=="USEFUL:") print $(i+1)}')
    PFUSELESS=$(echo "$PF" | awk '{for(i=1;i<=NF;i++) if($i=="USELESS:") print $(i+1)}')

    PFREQ=${PFREQ:-0}
    PFISS=${PFISS:-0}
    PFUSE=${PFUSE:-0}
    PFUSELESS=${PFUSELESS:-0}

    # ----------------------------
    # Cache Stats
    # ----------------------------
    L2MISS=$(grep "cpu0->cpu0_L2C TOTAL" "$LOG" | awk '{for(i=1;i<=NF;i++) if($i=="MISS:") print $(i+1)}')
    LLCMISS=$(grep "cpu0->LLC TOTAL" "$LOG" | awk '{for(i=1;i<=NF;i++) if($i=="MISS:") print $(i+1)}')

    LATL2=$(grep "cpu0->cpu0_L2C AVERAGE MISS LATENCY" "$LOG" | awk '{print $(NF-1)}')
    LATLLC=$(grep "cpu0->LLC AVERAGE MISS LATENCY" "$LOG" | awk '{print $(NF-1)}')

    L2MISS=${L2MISS:-0}
    LLCMISS=${LLCMISS:-0}
    LATL2=${LATL2:-0}
    LATLLC=${LATLLC:-0}

    # ----------------------------
    # DRAM Stats
    # ----------------------------
    RBH=$(grep "ROW_BUFFER_HIT:" "$LOG" | head -1 | awk '{print $3}')
    RBM=$(grep "ROW_BUFFER_MISS:" "$LOG" | head -1 | awk '{print $2}')
    CONG=$(grep "AVG DBUS CONGESTED CYCLE:" "$LOG" | awk '{print $5}')

    RBH=${RBH:-0}
    RBM=${RBM:-0}
    CONG=${CONG:-0}

    # ----------------------------
    # Derived Metrics
    # ----------------------------

    ACC=$(awk -v a="$PFUSE" -v b="$PFISS" 'BEGIN{if(b>0) printf "%.2f",100*a/b; else print 0}')
    POL=$(awk -v a="$PFUSELESS" -v b="$PFISS" 'BEGIN{if(b>0) printf "%.2f",100*a/b; else print 0}')
    COV=$(awk -v a="$PFUSE" -v b="$L2MISS" 'BEGIN{if(b>0) printf "%.2f",100*a/b; else print 0}')
    CPI=$(awk -v c="$CYC" -v i="$INS" 'BEGIN{if(i>0) printf "%.3f",c/i; else print 0}')

    ROWHIT=$(awk -v h="$RBH" -v m="$RBM" 'BEGIN{t=h+m; if(t>0) printf "%.2f",100*h/t; else print 0}')

    # Better Power Proxy
    POWER=$(awk -v pf="$PFISS" -v us="$PFUSELESS" -v llc="$LLCMISS" -v cg="$CONG" '
        BEGIN{
        printf "%.3f", 1 + pf*0.00004 + us*0.00010 + llc*0.000002 + cg*0.02
        }')

    # Energy Proxy
    ENERGY=$(awk -v p="$POWER" -v cyc="$CYC" '
        BEGIN{
        printf "%.3f", p * cyc / 100000
        }')

    # Efficiency
    EFF=$(awk -v ipc="$IPC" -v p="$POWER" '
        BEGIN{
        if(p>0) printf "%.3f", ipc/p; else print 0
        }')

    # Memory Pressure
    MPRESS=$(awk -v m="$RBM" -v c="$CONG" '
        BEGIN{
        printf "%.3f", m*0.001 + c*0.1
        }')

    {
        echo "=================================================="
        echo "Trace File              : $NAME"
        echo "Warmup Instructions     : $WARM"
        echo "Simulation Instructions : $SIM"
        echo "=================================================="

        echo ""
        echo "SYSTEM PERFORMANCE"
        echo "IPC                     : $IPC"
        echo "CPI                     : $CPI"
        echo "Instructions            : $INS"
        echo "Cycles                  : $CYC"

        echo ""
        echo "PREFETCH PERFORMANCE"
        echo "Prefetch Requested      : $PFREQ"
        echo "Prefetch Issued         : $PFISS"
        echo "Useful Prefetches       : $PFUSE"
        echo "Useless Prefetches      : $PFUSELESS"
        echo "Accuracy (%)            : $ACC"
        echo "Pollution (%)           : $POL"
        echo "Coverage (%)            : $COV"

        echo ""
        echo "CACHE BEHAVIOR"
        echo "L2 Misses               : $L2MISS"
        echo "LLC Misses              : $LLCMISS"
        echo "L2 Miss Latency         : $LATL2"
        echo "LLC Miss Latency        : $LATLLC"

        echo ""
        echo "MEMORY SYSTEM"
        echo "Row Buffer Hit Rate (%) : $ROWHIT"
        echo "Bus Congestion          : $CONG"
        echo "Memory Pressure Score   : $MPRESS"

        echo ""
        echo "POWER / ENERGY (Proxy)"
        echo "Power Score             : $POWER"
        echo "Energy Score            : $ENERGY"
        echo "Perf/Watt Score         : $EFF"

        echo ""
        echo "NOTES"
        echo "- Miss reduction requires baseline comparison"
        echo "- Latency unaffected requires baseline comparison"
        echo "- Power/Energy are architecture-level proxies"
        echo "=================================================="
    } > "$REP"

    {
        echo "$NAME | IPC=$IPC | Acc=$ACC% | Power=$POWER | Energy=$ENERGY | Eff=$EFF"
    } >> "$MASTER"

    echo "Saved -> $REP"
done

echo "All traces completed."
echo "Master summary -> $MASTER"
