import folium
from sklearn.datasets import fetch_california_housing

# 1. 데이터 로드
data = fetch_california_housing()
X = data.data
feature_names = data.feature_names

# 위도, 경도, 중위소득 추출
lat = X[:, feature_names.index("Latitude")]
lon = X[:, feature_names.index("Longitude")]
income = X[:, feature_names.index("MedInc")]

# 2. 지도 생성
m = folium.Map(location=[36.7783, -119.4179], zoom_start=6, tiles='cartodbpositron')

# 3. CircleMarker 추가 (소득 기준 색상 분류)
for i in range(len(lat)):
    inc = income[i]
    color = (
        'blue' if inc < 3 else
        'orange' if inc < 6 else
        'red'
    )
    folium.CircleMarker(
        location=[lat[i], lon[i]],
        radius=2.2,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        color=None,
        popup=f"Income: ${inc*10000:.0f}"
    ).add_to(m)

# 4. 범례 추가 (HTML 형식)
legend_html = '''
<div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 160px; height: 120px; 
     background-color: white; 
     border:2px solid grey; 
     z-index:9999;
     font-size:14px;
     padding: 10px;
     box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
     ">
     <b>Median Income</b><br>
     <i style="background:blue;width:10px;height:10px;display:inline-block;margin-right:5px;"></i> < $30,000<br>
     <i style="background:orange;width:10px;height:10px;display:inline-block;margin-right:5px;"></i> $30,000–$60,000<br>
     <i style="background:red;width:10px;height:10px;display:inline-block;margin-right:5px;"></i> > $60,000
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# 5. HTML 파일로 저장
m.save("california_income_full_map.html")
print("지도 생성 완료: california_income_full_map.html")
