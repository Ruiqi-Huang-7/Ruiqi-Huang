#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Data Analysis Script
气候数据分析脚本

This script analyzes climate data to identify trends, patterns, and potential project opportunities.
该脚本分析气候数据以识别趋势、模式和潜在的项目机会。
"""

# Import necessary libraries
# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import json

# Set visualization styles
# 设置可视化样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_and_preprocess_data(file_path):
    """
    Load and preprocess climate data
    加载和预处理气候数据
    
    Args:
        file_path (str): Path to the CSV data file
                         CSV数据文件路径
                         
    Returns:
        pandas.DataFrame: Preprocessed climate data
                          预处理后的气候数据
    """
    print("Loading and preprocessing data...")
    print("正在加载和预处理数据...")
    
    # Load data from CSV file
    # 从CSV文件加载数据
    df = pd.read_csv(file_path)
    
    # Convert time column to datetime format
    # 将时间列转换为日期时间格式
    df['time'] = pd.to_datetime(df['time'])
    
    # Extract time components for analysis
    # 提取时间组件用于分析
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    
    # Define seasons based on months
    # 根据月份定义季节
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    # Calculate wind speed from U and V components
    # 从U和V分量计算风速
    df['wind_speed'] = np.sqrt(df['UBOT']**2 + df['VBOT']**2)
    
    # Convert temperature from Kelvin to Celsius for easier interpretation
    # 将温度从开尔文转换为摄氏度以便于解释
    df['TREFHT_C'] = df['TREFHT'] - 273.15
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"数据加载成功。形状: {df.shape}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"时间范围: {df['time'].min()} 至 {df['time'].max()}")
    
    return df

def analyze_climate_trends(df):
    """
    Analyze long-term climate trends
    分析长期气候趋势
    
    Args:
        df (pandas.DataFrame): Climate data
                               气候数据
                               
    Returns:
        dict: Analysis results
              分析结果
    """
    print("\nAnalyzing climate trends...")
    print("\n正在分析气候趋势...")
    
    # Calculate yearly averages for key variables
    # 计算关键变量的年平均值
    yearly_avg = df.groupby('year').agg({
        'TREFHT': 'mean',
        'TREFHT_C': 'mean',
        'FSNS': 'mean',
        'FLNS': 'mean',
        'QBOT': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    # Calculate temperature trend using linear regression
    # 使用线性回归计算温度趋势
    X = yearly_avg[['year']]
    y = yearly_avg['TREFHT_C']
    model = LinearRegression()
    model.fit(X, y)
    temp_trend = model.coef_[0] * 10  # Trend per decade
                                      # 每十年的趋势
    
    print(f"Temperature trend: {temp_trend:.2f} °C per decade")
    print(f"温度趋势: 每十年 {temp_trend:.2f} °C")
    
    # Analyze seasonal patterns
    # 分析季节性模式
    seasonal_avg = df.groupby('season').agg({
        'TREFHT_C': 'mean',
        'FSNS': 'mean',
        'FLNS': 'mean',
        'QBOT': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    print("\nSeasonal temperature averages (°C):")
    print("\n季节性温度平均值 (°C):")
    for _, row in seasonal_avg.iterrows():
        print(f"{row['season']}: {row['TREFHT_C']:.1f}°C")
    
    # Analyze monthly patterns
    # 分析月度模式
    monthly_avg = df.groupby('month')['TREFHT_C'].mean().reset_index()
    
    return {
        'yearly_data': yearly_avg,
        'temperature_trend': temp_trend,
        'seasonal_data': seasonal_avg,
        'monthly_data': monthly_avg
    }

def identify_extreme_events(df):
    """
    Identify potential extreme weather events
    识别潜在的极端天气事件
    
    Args:
        df (pandas.DataFrame): Climate data
                               气候数据
                               
    Returns:
        dict: Extreme events analysis
              极端事件分析
    """
    print("\nIdentifying extreme weather events...")
    print("\n正在识别极端天气事件...")
    
    # Calculate temperature percentiles
    # 计算温度百分位数
    temp_95th = df['TREFHT_C'].quantile(0.95)
    temp_5th = df['TREFHT_C'].quantile(0.05)
    
    # Identify extreme heat and cold days
    # 识别极端高温和低温天数
    extreme_heat_days = df[df['TREFHT_C'] >= temp_95th]
    extreme_cold_days = df[df['TREFHT_C'] <= temp_5th]
    
    # Analyze extreme wind events
    # 分析极端风事件
    wind_95th = df['wind_speed'].quantile(0.95)
    extreme_wind_days = df[df['wind_speed'] >= wind_95th]
    
    print(f"Extreme heat days (above {temp_95th:.1f}°C): {len(extreme_heat_days)}")
    print(f"极端高温天数 (高于 {temp_95th:.1f}°C): {len(extreme_heat_days)}")
    print(f"Extreme cold days (below {temp_5th:.1f}°C): {len(extreme_cold_days)}")
    print(f"极端低温天数 (低于 {temp_5th:.1f}°C): {len(extreme_cold_days)}")
    print(f"Extreme wind days (above {wind_95th:.1f} m/s): {len(extreme_wind_days)}")
    print(f"极端大风天数 (高于 {wind_95th:.1f} m/s): {len(extreme_wind_days)}")
    
    return {
        'extreme_heat': extreme_heat_days,
        'extreme_cold': extreme_cold_days,
        'extreme_wind': extreme_wind_days,
        'temp_thresholds': {'heat': temp_95th, 'cold': temp_5th},
        'wind_threshold': wind_95th
    }

def assess_renewable_potential(df):
    """
    Assess potential for renewable energy generation
    评估可再生能源发电潜力
    
    Args:
        df (pandas.DataFrame): Climate data
                               气候数据
                               
    Returns:
        dict: Renewable energy potential assessment
              可再生能源潜力评估
    """
    print("\nAssessing renewable energy potential...")
    print("\n正在评估可再生能源潜力...")
    
    # Solar potential based on net shortwave radiation
    # 基于净短波辐射的太阳能潜力
    solar_potential = df.groupby('month')['FSNS'].mean().reset_index()
    solar_potential['solar_rank'] = solar_potential['FSNS'].rank(ascending=False)
    
    # Wind potential based on wind speed
    # 基于风速的风能潜力
    wind_potential = df.groupby('month')['wind_speed'].mean().reset_index()
    wind_potential['wind_rank'] = wind_potential['wind_speed'].rank(ascending=False)
    
    print("Top 3 months for solar potential:")
    print("太阳能潜力排名前3的月份:")
    top_solar = solar_potential.nsmallest(3, 'solar_rank')
    for _, row in top_solar.iterrows():
        print(f"Month {int(row['month'])}: {row['FSNS']:.1f} W/m²")
    
    print("\nTop 3 months for wind potential:")
    print("\n风能潜力排名前3的月份:")
    top_wind = wind_potential.nsmallest(3, 'wind_rank')
    for _, row in top_wind.iterrows():
        print(f"Month {int(row['month'])}: {row['wind_speed']:.1f} m/s")
    
    return {
        'solar_potential': solar_potential,
        'wind_potential': wind_potential
    }

def prepare_visualization_data(df, analysis_results):
    """
    Prepare data for visualization
    准备可视化数据
    
    Args:
        df (pandas.DataFrame): Original climate data
                               原始气候数据
        analysis_results (dict): Results from climate analysis
                                 气候分析结果
                                 
    Returns:
        dict: Data ready for visualization
              准备好用于可视化的数据
    """
    print("\nPreparing visualization data...")
    print("\n正在准备可视化数据...")
    
    # Sample data for display (10 random points)
    # 用于显示的样本数据（10个随机点）
    sample_data = df.sample(n=min(10, len(df))).copy()
    sample_data['time_str'] = sample_data['time'].dt.strftime('%Y-%m-%d')
    
    # Yearly trend data
    # 年际趋势数据
    yearly_data = analysis_results['yearly_data'].copy()
    
    # Seasonal data
    # 季节性数据
    seasonal_data = analysis_results['seasonal_data'].copy()
    
    # Monthly temperature data
    # 月度温度数据
    monthly_temp = analysis_results['monthly_data'].copy()
    
    # Renewable potential data
    # 可再生能源潜力数据
    renewable_data = {
        'solar': analysis_results['renewable']['solar_potential'].to_dict('records'),
        'wind': analysis_results['renewable']['wind_potential'].to_dict('records')
    }
    
    # Extreme events statistics
    # 极端事件统计
    extreme_stats = {
        'heat_days': len(analysis_results['extreme']['extreme_heat']),
        'cold_days': len(analysis_results['extreme']['extreme_cold']),
        'wind_days': len(analysis_results['extreme']['extreme_wind']),
        'heat_threshold': analysis_results['extreme']['temp_thresholds']['heat'],
        'cold_threshold': analysis_results['extreme']['temp_thresholds']['cold'],
        'wind_threshold': analysis_results['extreme']['wind_threshold']
    }
    
    visualization_data = {
        'sample_data': sample_data[['time_str', 'TREFHT_C', 'FSNS', 'FLNS', 'QBOT', 'wind_speed', 'season']].to_dict('records'),
        'yearly_trend': yearly_data.to_dict('records'),
        'seasonal_avg': seasonal_data.to_dict('records'),
        'monthly_temp': monthly_temp.to_dict('records'),
        'renewable_potential': renewable_data,
        'extreme_events': extreme_stats,
        'temperature_trend': analysis_results['temperature_trend']
    }
    
    print("Visualization data prepared successfully")
    print("可视化数据准备成功")
    
    return visualization_data

def generate_project_recommendations(analysis_results):
    """
    Generate project recommendations based on data analysis
    基于数据分析生成项目建议
    
    Args:
        analysis_results (dict): Results from climate analysis
                                 气候分析结果
                                 
    Returns:
        list: Project recommendations
              项目建议
    """
    print("\nGenerating project recommendations...")
    print("\n正在生成项目建议...")
    
    recommendations = []
    
    # Recommendation 1: Climate Change Trend Analysis and Prediction
    # 建议1：气候变化趋势分析与预测
    recommendations.append({
        'title': 'Climate Change Trend Analysis and Prediction Model',
        'title_cn': '气候变化趋势分析与预测模型',
        'description': 'Develop a comprehensive model to analyze historical climate trends and predict future climate conditions based on the 75-year dataset.',
        'description_cn': '开发一个综合模型，基于75年的数据集分析历史气候趋势并预测未来气候条件。',
        'key_features': [
            'Time series analysis of temperature, radiation, and wind patterns',
            'Development of predictive models using machine learning algorithms',
            'Visualization dashboard for trend monitoring',
            'Uncertainty analysis of future predictions'
        ],
        'key_features_cn': [
            '温度、辐射和风模式的时间序列分析',
            '使用机器学习算法开发预测模型',
            '趋势监测可视化仪表板',
            '未来预测的不确定性分析'
        ],
        'technical_approach': 'Time series forecasting, regression analysis, LSTM neural networks',
        'technical_approach_cn': '时间序列预测、回归分析、LSTM神经网络'
    })
    
    # Recommendation 2: Extreme Weather Event Detection and Early Warning System
    # 建议2：极端天气事件检测与预警系统
    recommendations.append({
        'title': 'Extreme Weather Event Detection and Early Warning System',
        'title_cn': '极端天气事件检测与预警系统',
        'description': 'Create a system to identify and predict extreme weather events such as heatwaves, cold snaps, and high wind events.',
        'description_cn': '创建一个系统来识别和预测极端天气事件，如热浪、寒潮和大风事件。',
        'key_features': [
            'Real-time monitoring of climate parameters',
            'Anomaly detection algorithms for early warning',
            'Risk assessment framework for different event types',
            'Notification system for stakeholders'
        ],
        'key_features_cn': [
            '气候参数的实时监测',
            '用于早期预警的异常检测算法',
            '不同事件类型的风险评估框架',
            '利益相关者通知系统'
        ],
        'technical_approach': 'Anomaly detection, statistical process control, classification algorithms',
        'technical_approach_cn': '异常检测、统计过程控制、分类算法'
    })
    
    # Recommendation 3: Renewable Energy Potential Assessment Platform
    # 建议3：可再生能源潜力评估平台
    recommendations.append({
        'title': 'Renewable Energy Potential Assessment Platform',
        'title_cn': '可再生能源潜力评估平台',
        'description': 'Build a platform to evaluate solar and wind energy potential based on historical climate data patterns.',
        'description_cn': '构建一个基于历史气候数据模式评估太阳能和风能潜力的平台。',
        'key_features': [
            'Solar irradiance mapping and analysis',
            'Wind resource assessment and mapping',
            'Energy production forecasting models',
            'Economic viability analysis tools'
        ],
        'key_features_cn': [
            '太阳辐照度映射和分析',
            '风能资源评估和映射',
            '能源生产预测模型',
            '经济可行性分析工具'
        ],
        'technical_approach': 'Spatial analysis, energy conversion modeling, economic analysis',
        'technical_approach_cn': '空间分析、能源转换建模、经济分析'
    })
    
    # Recommendation 4: Climate Impact Analysis on Ecosystems/Agriculture
    # 建议4：气候变化对生态系统/农业的影响分析
    recommendations.append({
        'title': 'Climate Impact Analysis on Ecosystems and Agriculture',
        'title_cn': '气候变化对生态系统和农业的影响分析',
        'description': 'Analyze how changing climate patterns might impact local ecosystems, biodiversity, and agricultural productivity.',
        'description_cn': '分析气候变化模式如何影响当地生态系统、生物多样性和农业生产力。',
        'key_features': [
            'Correlation analysis between climate variables and ecosystem indicators',
            'Scenario modeling for different climate change trajectories',
            'Vulnerability assessment framework',
            'Adaptation strategy recommendations'
        ],
        'key_features_cn': [
            '气候变量与生态系统指标的相关性分析',
            '不同气候变化轨迹的情景建模',
            '脆弱性评估框架',
            '适应策略建议'
        ],
        'technical_approach': 'Correlation analysis, scenario planning, impact assessment methodologies',
        'technical_approach_cn': '相关性分析、情景规划、影响评估方法'
    })
    
    print(f"Generated {len(recommendations)} project recommendations")
    print(f"已生成 {len(recommendations)} 个项目建议")
    
    return recommendations

def main():
    """
    Main function to execute the climate data analysis workflow
    执行气候数据分析工作流程的主函数
    """
    print("Starting Climate Data Analysis Project...")
    print("开始气候数据分析项目...")
    
    try:
        # Step 1: Load and preprocess data
        # 步骤1：加载和预处理数据
        df = load_and_preprocess_data('project_1.csv')
        
        # Step 2: Analyze climate trends
        # 步骤2：分析气候趋势
        trend_results = analyze_climate_trends(df)
        
        # Step 3: Identify extreme weather events
        # 步骤3：识别极端天气事件
        extreme_results = identify_extreme_events(df)
        
        # Step 4: Assess renewable energy potential
        # 步骤4：评估可再生能源潜力
        renewable_results = assess_renewable_potential(df)
        
        # Combine analysis results
        # 合并分析结果
        analysis_results = {
            'yearly_data': trend_results['yearly_data'],
            'temperature_trend': trend_results['temperature_trend'],
            'seasonal_data': trend_results['seasonal_data'],
            'monthly_data': trend_results['monthly_data'],
            'extreme': extreme_results,
            'renewable': renewable_results
        }
        
        # Step 5: Prepare visualization data
        # 步骤5：准备可视化数据
        viz_data = prepare_visualization_data(df, analysis_results)
        
        # Step 6: Generate project recommendations
        # 步骤6：生成项目建议
        recommendations = generate_project_recommendations(analysis_results)
        
        # Save visualization data to JSON file
        # 将可视化数据保存到JSON文件
        with open('climate_visualization_data.json', 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        # Save recommendations to JSON file
        # 将建议保存到JSON文件
        with open('project_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print("\nAnalysis completed successfully!")
        print("Output files:")
        print("- climate_visualization_data.json: Data for visualization")
        print("- project_recommendations.json: Project recommendations")
        print("\n分析成功完成！")
        print("输出文件:")
        print("- climate_visualization_data.json: 可视化数据")
        print("- project_recommendations.json: 项目建议")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()