import sqlite3
import matplotlib.pyplot as plt

# connect to DB
db_path = "report_cuda_nvtx.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# fetching data
query = """
SELECT text, start, end, (end - start)
FROM NVTX_EVENTS
WHERE text IS NOT NULL
ORDER BY start ASC
"""

try:
    cursor.execute(query)
    rows = cursor.fetchall()

    if not rows:
        print("No Data fetching")
        conn.close()
        exit()

    
    min_start_ns = rows[0][1]
    data_for_plot = []

    print(f"{'Message':<30} | {'Start (ms)':<15} | {'Duration (ms)':<15}")
    print("=" * 70)

    for row in rows:
        msg = str(row[0])
        start_ns = row[1]
        end_ns = row[2]
        duration_ns = row[3]

        start_ms = (start_ns - min_start_ns) / 1e6
        duration_ms = duration_ns / 1e6
        
        # 계층(Level) 및 색상
        if "Program" in msg:
            y_pos = 3; 
            color = 'tab:gray'
        elif "Step" in msg:
            y_pos = 2; 
            color = 'tab:blue'
        elif "Matmul" in msg:
            y_pos = 1; 
            color = 'tab:green'
        elif "Sync" in msg:
            y_pos = 1; 
            color = 'tab:red'
        else:
            y_pos = 0; 
            color = 'tab:purple'

        data_for_plot.append((msg, start_ms, duration_ms, y_pos, color))
        
        # 터미널 로그 출력
        indent = "  " * (3 - y_pos)
        print(f"{indent}{msg:<30} | {start_ms:<15.3f} | {duration_ms:<15.3f}")

    # Ploting 
    fig, ax = plt.subplots(figsize=(14, 6)) # 크기 약간 키움

    added_legend = set()
    for msg, start, duration, y, color in data_for_plot:
        lbl = msg.split()[0] if "Step" in msg else msg
        plot_label = lbl if lbl not in added_legend else ""
        added_legend.add(lbl)

        ax.barh(y=y, width=duration, left=start, height=0.6, color=color, label=plot_label, edgecolor='black', alpha=0.8)

        if duration > 1.0: # 아주 짧은건 글자 생략
            ax.text(start + duration/2, y, msg, ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_xlabel("Time (ms)")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Operations", "Steps", "Program"])
    ax.set_title("NVTX Timeline Analysis")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    
    # ==========================================
    # ★ 수정된 부분: 창 띄우기 대신 파일 저장
    # ==========================================
    output_file = "nvtx_timeline.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ 그래프 저장 완료: {output_file}")
    print("왼쪽 파일 탐색기에서 nvtx_timeline.png 파일을 열어보세요.")

except sqlite3.Error as e:
    print(f"SQL 에러: {e}")

finally:
    conn.close()