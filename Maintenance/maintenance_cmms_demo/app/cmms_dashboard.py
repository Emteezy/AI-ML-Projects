"""
CMMS Streamlit Dashboard
Comprehensive maintenance management interface
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import httpx
from typing import Dict, List

# Configuration
API_URL = "http://localhost:8003"

# Page config
st.set_page_config(
    page_title="CMMS Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def call_api(method: str, endpoint: str, data: dict = None):
    """Call CMMS API"""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = httpx.get(url, timeout=5.0)
        elif method == "POST":
            response = httpx.post(url, json=data, timeout=5.0)
        elif method == "PATCH":
            response = httpx.patch(url, json=data, timeout=5.0)
        else:
            return None
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.kpi-value {
    font-size: 36px;
    font-weight: bold;
}
.kpi-label {
    font-size: 14px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 CMMS Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Assets", "PM Scheduler", "Work Orders", "Downtime", "Spares", "RCA", "Reports"]
)

st.sidebar.markdown("---")
st.sidebar.info("Maintenance Management System")

# Main content
st.title("CMMS - Maintenance Management System")

# ========== DASHBOARD PAGE ==========
if page == "Dashboard":
    st.header("📊 KPI Dashboard")
    
    # Get KPIs
    kpis = call_api("GET", "/kpis?days=30")
    
    if kpis:
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MTTR (hours)", f"{kpis.get('mttr_hours', 0):.1f}")
            st.caption("Mean Time To Repair")
        
        with col2:
            st.metric("MTBF (hours)", f"{kpis.get('mtbf_hours', 0):.1f}")
            st.caption("Mean Time Between Failures")
        
        with col3:
            st.metric("Downtime (hours)", f"{kpis.get('total_downtime_hours', 0):.1f}")
            st.caption("Last 30 days")
        
        with col4:
            st.metric("PM Compliance", f"{kpis.get('pm_compliance_percent', 0):.1f}%")
            st.caption("On-time completion")
        
        st.markdown("---")
        
        # Top failure modes
        st.subheader("Top Failure Modes")
        failure_modes = kpis.get('top_failure_modes', [])
        
        if failure_modes:
            df = pd.DataFrame(failure_modes)
            
            fig = px.bar(
                df,
                x='failure_mode',
                y='count',
                title="Failure Mode Frequency",
                labels={'failure_mode': 'Failure Mode', 'count': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No failure data available")
        
        # Recent work orders
        st.subheader("Recent Work Orders")
        work_orders = call_api("GET", "/work-orders")
        if work_orders:
            df_wo = pd.DataFrame(work_orders)
            df_wo = df_wo.sort_values('created_at', ascending=False).head(10)
            st.dataframe(
                df_wo[['wo_number', 'title', 'wo_type', 'status', 'priority', 'assigned_to']],
                use_container_width=True
            )

# ========== ASSETS PAGE ==========
elif page == "Assets":
    st.header("🏭 Asset Management")
    
    tab1, tab2 = st.tabs(["Asset List", "Add New Asset"])
    
    with tab1:
        assets = call_api("GET", "/assets")
        if assets:
            df = pd.DataFrame(assets)
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                category_filter = st.selectbox(
                    "Filter by Category",
                    ["All"] + list(df['category'].unique())
                )
            with col2:
                criticality_filter = st.selectbox(
                    "Filter by Criticality",
                    ["All"] + list(df['criticality'].unique())
                )
            
            # Apply filters
            filtered_df = df.copy()
            if category_filter != "All":
                filtered_df = filtered_df[filtered_df['category'] == category_filter]
            if criticality_filter != "All":
                filtered_df = filtered_df[filtered_df['criticality'] == criticality_filter]
            
            # Display
            st.dataframe(
                filtered_df[['asset_tag', 'name', 'category', 'location', 'criticality', 'status']],
                use_container_width=True
            )
            
            # Asset details
            st.subheader("Asset Details")
            selected_asset = st.selectbox(
                "Select Asset",
                options=filtered_df['asset_tag'].tolist()
            )
            
            if selected_asset:
                asset_data = filtered_df[filtered_df['asset_tag'] == selected_asset].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Asset Tag:**", asset_data['asset_tag'])
                    st.write("**Name:**", asset_data['name'])
                    st.write("**Category:**", asset_data['category'])
                    st.write("**Location:**", asset_data['location'])
                
                with col2:
                    st.write("**Manufacturer:**", asset_data['manufacturer'])
                    st.write("**Model:**", asset_data['model'])
                    st.write("**Criticality:**", asset_data['criticality'])
                    st.write("**Status:**", asset_data['status'])
        else:
            st.info("No assets found")
    
    with tab2:
        st.subheader("Add New Asset")
        
        with st.form("new_asset_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                asset_tag = st.text_input("Asset Tag*")
                name = st.text_input("Name*")
                category = st.selectbox("Category", ["Motor", "Pump", "Conveyor", "Compressor", "Robot"])
                location = st.text_input("Location")
            
            with col2:
                manufacturer = st.text_input("Manufacturer")
                model = st.text_input("Model")
                serial_number = st.text_input("Serial Number")
                criticality = st.selectbox("Criticality", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            
            notes = st.text_area("Notes")
            
            submitted = st.form_submit_button("Create Asset")
            
            if submitted:
                if asset_tag and name:
                    data = {
                        "asset_tag": asset_tag,
                        "name": name,
                        "category": category,
                        "location": location,
                        "manufacturer": manufacturer,
                        "model": model,
                        "serial_number": serial_number,
                        "criticality": criticality,
                        "notes": notes
                    }
                    result = call_api("POST", "/assets", data)
                    if result:
                        st.success("Asset created successfully!")
                        st.rerun()
                else:
                    st.error("Asset Tag and Name are required")

# ========== PM SCHEDULER PAGE ==========
elif page == "PM Scheduler":
    st.header("📅 PM Scheduler")
    
    tab1, tab2 = st.tabs(["PM Plans", "Create PM Plan"])
    
    with tab1:
        pm_plans = call_api("GET", "/pm-plans")
        if pm_plans:
            df = pd.DataFrame(pm_plans)
            
            # Show upcoming PMs
            st.subheader("Upcoming PM Tasks")
            df['next_due'] = pd.to_datetime(df['next_due'])
            upcoming = df[df['active'] == True].sort_values('next_due')
            
            for _, plan in upcoming.head(10).iterrows():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                with col1:
                    st.write(f"**{plan['plan_name']}**")
                with col2:
                    days_until = (plan['next_due'] - datetime.now()).days
                    if days_until < 0:
                        st.error(f"Overdue by {abs(days_until)} days")
                    elif days_until < 7:
                        st.warning(f"Due in {days_until} days")
                    else:
                        st.info(f"Due in {days_until} days")
                with col3:
                    st.write(f"Every {plan['frequency_days']} days")
                with col4:
                    st.write(f"{plan['estimated_hours']:.1f} hours")
                
                st.markdown("---")
        else:
            st.info("No PM plans found")
    
    with tab2:
        st.subheader("Create PM Plan")
        
        assets = call_api("GET", "/assets")
        if assets:
            asset_options = {f"{a['asset_tag']} - {a['name']}": a['id'] for a in assets}
            
            with st.form("new_pm_form"):
                selected_asset = st.selectbox("Asset", list(asset_options.keys()))
                plan_name = st.text_input("Plan Name*")
                description = st.text_area("Description")
                frequency_days = st.number_input("Frequency (days)*", min_value=1, value=30)
                estimated_hours = st.number_input("Estimated Hours", min_value=0.0, value=2.0, step=0.5)
                checklist = st.text_area("Checklist Items (one per line)")
                
                submitted = st.form_submit_button("Create PM Plan")
                
                if submitted and plan_name:
                    data = {
                        "asset_id": asset_options[selected_asset],
                        "plan_name": plan_name,
                        "description": description,
                        "frequency_days": frequency_days,
                        "estimated_hours": estimated_hours,
                        "checklist_items": checklist
                    }
                    result = call_api("POST", "/pm-plans", data)
                    if result:
                        st.success("PM Plan created!")
                        st.rerun()

# ========== WORK ORDERS PAGE ==========
elif page == "Work Orders":
    st.header("📋 Work Orders")
    
    tab1, tab2 = st.tabs(["Work Order List", "Create Work Order"])
    
    with tab1:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Status", ["All", "OPEN", "ASSIGNED", "IN_PROGRESS", "COMPLETED", "CLOSED"])
        with col2:
            type_filter = st.selectbox("Type", ["All", "PM", "CM"])
        with col3:
            st.write("")  # Spacing
        
        # Get work orders
        params = ""
        if status_filter != "All":
            params += f"?status={status_filter}"
        if type_filter != "All":
            params += f"{'&' if '?' in params else '?'}wo_type={type_filter}"
        
        work_orders = call_api("GET", f"/work-orders{params}")
        
        if work_orders:
            df = pd.DataFrame(work_orders)
            
            st.dataframe(
                df[['wo_number', 'title', 'wo_type', 'status', 'priority', 'assigned_to', 'created_at']],
                use_container_width=True
            )
            
            # Work order details
            st.subheader("Work Order Details")
            selected_wo = st.selectbox("Select Work Order", df['wo_number'].tolist())
            
            if selected_wo:
                wo_data = df[df['wo_number'] == selected_wo].iloc[0]
                wo_id = wo_data['id']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**WO Number:**", wo_data['wo_number'])
                    st.write("**Type:**", wo_data['wo_type'])
                    st.write("**Priority:**", wo_data['priority'])
                    st.write("**Status:**", wo_data['status'])
                    st.write("**Title:**", wo_data['title'])
                
                with col2:
                    st.write("**Assigned To:**", wo_data['assigned_to'])
                    st.write("**Created:**", wo_data['created_at'])
                    st.write("**Estimated Hours:**", wo_data['estimated_hours'])
                    if wo_data['actual_hours']:
                        st.write("**Actual Hours:**", wo_data['actual_hours'])
                
                st.write("**Description:**", wo_data['description'])
                
                # Update status
                st.subheader("Update Work Order")
                new_status = st.selectbox("Change Status", ["OPEN", "ASSIGNED", "IN_PROGRESS", "COMPLETED", "CLOSED"])
                actual_hours = st.number_input("Actual Hours", min_value=0.0, value=0.0, step=0.5)
                resolution = st.text_area("Resolution Notes")
                
                if st.button("Update Work Order"):
                    update_data = {
                        "status": new_status,
                        "actual_hours": actual_hours if actual_hours > 0 else None,
                        "resolution_notes": resolution if resolution else None
                    }
                    if new_status == "COMPLETED":
                        update_data["completed_at"] = datetime.now().isoformat()
                    
                    result = call_api("PATCH", f"/work-orders/{wo_id}", update_data)
                    if result:
                        st.success("Work order updated!")
                        st.rerun()
        else:
            st.info("No work orders found")
    
    with tab2:
        st.subheader("Create Work Order")
        
        assets = call_api("GET", "/assets")
        if assets:
            asset_options = {f"{a['asset_tag']} - {a['name']}": a['id'] for a in assets}
            
            with st.form("new_wo_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_asset = st.selectbox("Asset*", list(asset_options.keys()))
                    wo_type = st.selectbox("Type*", ["PM", "CM"])
                    priority = st.selectbox("Priority", ["LOW", "MEDIUM", "HIGH", "URGENT"])
                
                with col2:
                    title = st.text_input("Title*")
                    assigned_to = st.text_input("Assign To")
                    estimated_hours = st.number_input("Estimated Hours", min_value=0.0, value=2.0, step=0.5)
                
                description = st.text_area("Description")
                scheduled_date = st.date_input("Scheduled Date")
                
                submitted = st.form_submit_button("Create Work Order")
                
                if submitted and title:
                    data = {
                        "asset_id": asset_options[selected_asset],
                        "wo_type": wo_type,
                        "priority": priority,
                        "title": title,
                        "description": description,
                        "assigned_to": assigned_to,
                        "created_by": "Dashboard User",
                        "estimated_hours": estimated_hours,
                        "scheduled_date": datetime.combine(scheduled_date, datetime.min.time()).isoformat()
                    }
                    result = call_api("POST", "/work-orders", data)
                    if result:
                        st.success(f"Work Order {result['wo_number']} created!")
                        st.rerun()

# ========== DOWNTIME PAGE ==========
elif page == "Downtime":
    st.header("⏱️ Downtime Tracking")
    
    tab1, tab2 = st.tabs(["Downtime Incidents", "Record Incident"])
    
    with tab1:
        incidents = call_api("GET", "/downtime")
        if incidents:
            df = pd.DataFrame(incidents)
            df['start_time'] = pd.to_datetime(df['start_time'])
            df = df.sort_values('start_time', ascending=False)
            
            st.dataframe(
                df[['incident_number', 'failure_mode', 'severity', 'downtime_hours', 'start_time']],
                use_container_width=True
            )
            
            # Downtime by severity
            st.subheader("Downtime by Severity")
            severity_summary = df.groupby('severity')['downtime_hours'].sum().reset_index()
            
            fig = px.pie(
                severity_summary,
                values='downtime_hours',
                names='severity',
                title="Total Downtime by Severity"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No downtime incidents recorded")
    
    with tab2:
        st.subheader("Record Downtime Incident")
        
        assets = call_api("GET", "/assets")
        if assets:
            asset_options = {f"{a['asset_tag']} - {a['name']}": a['id'] for a in assets}
            
            with st.form("new_incident_form"):
                selected_asset = st.selectbox("Asset*", list(asset_options.keys()))
                
                col1, col2 = st.columns(2)
                with col1:
                    start_time = st.datetime_input("Start Time*", value=datetime.now())
                    failure_mode = st.text_input("Failure Mode")
                with col2:
                    end_time = st.datetime_input("End Time (optional)")
                    severity = st.selectbox("Severity", ["MINOR", "MAJOR", "CRITICAL"])
                
                description = st.text_area("Description")
                immediate_action = st.text_area("Immediate Action Taken")
                
                submitted = st.form_submit_button("Record Incident")
                
                if submitted:
                    # Generate incident number
                    existing = call_api("GET", "/downtime")
                    incident_number = f"INC-{len(existing) + 1:04d}"
                    
                    data = {
                        "asset_id": asset_options[selected_asset],
                        "incident_number": incident_number,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat() if end_time else None,
                        "failure_mode": failure_mode,
                        "severity": severity,
                        "description": description,
                        "immediate_action": immediate_action
                    }
                    result = call_api("POST", "/downtime", data)
                    if result:
                        st.success(f"Incident {incident_number} recorded!")
                        st.rerun()

# ========== SPARES PAGE ==========
elif page == "Spares":
    st.header("📦 Spare Parts Inventory")
    
    tab1, tab2 = st.tabs(["Inventory", "Add Part"])
    
    with tab1:
        parts = call_api("GET", "/spare-parts")
        if parts:
            df = pd.DataFrame(parts)
            
            # Low stock warning
            low_stock = df[df['quantity_on_hand'] <= df['min_quantity']]
            if not low_stock.empty:
                st.warning(f"⚠️ {len(low_stock)} parts below minimum quantity!")
                with st.expander("Show Low Stock Parts"):
                    st.dataframe(
                        low_stock[['part_number', 'description', 'quantity_on_hand', 'min_quantity']],
                        use_container_width=True
                    )
            
            st.dataframe(
                df[['part_number', 'description', 'category', 'quantity_on_hand', 'min_quantity', 'unit_cost']],
                use_container_width=True
            )
        else:
            st.info("No spare parts in inventory")
    
    with tab2:
        st.subheader("Add Spare Part")
        
        with st.form("new_part_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                part_number = st.text_input("Part Number*")
                description = st.text_input("Description*")
                category = st.text_input("Category")
                quantity = st.number_input("Quantity on Hand", min_value=0, value=0)
            
            with col2:
                min_quantity = st.number_input("Minimum Quantity", min_value=0, value=0)
                unit_cost = st.number_input("Unit Cost", min_value=0.0, value=0.0, step=0.01)
                location = st.text_input("Location")
                supplier = st.text_input("Supplier")
            
            submitted = st.form_submit_button("Add Part")
            
            if submitted and part_number and description:
                data = {
                    "part_number": part_number,
                    "description": description,
                    "category": category,
                    "quantity_on_hand": quantity,
                    "min_quantity": min_quantity,
                    "unit_cost": unit_cost,
                    "location": location,
                    "supplier": supplier
                }
                result = call_api("POST", "/spare-parts", data)
                if result:
                    st.success("Part added to inventory!")
                    st.rerun()

# ========== RCA PAGE ==========
elif page == "RCA":
    st.header("🔍 Root Cause Analysis")
    
    incidents = call_api("GET", "/downtime")
    if incidents:
        incident_options = {f"{i['incident_number']} - {i['failure_mode']}": i['id'] for i in incidents}
        
        selected_incident = st.selectbox("Select Incident", list(incident_options.keys()))
        incident_id = incident_options[selected_incident]
        
        # Check if RCA exists
        existing_rca = call_api("GET", f"/rca/{incident_id}")
        
        if existing_rca:
            st.success("✓ RCA exists for this incident")
            
            # Display RCA
            st.subheader("5-Why Analysis")
            st.write("**Why 1:**", existing_rca.get('why_1', 'N/A'))
            st.write("**Why 2:**", existing_rca.get('why_2', 'N/A'))
            st.write("**Why 3:**", existing_rca.get('why_3', 'N/A'))
            st.write("**Why 4:**", existing_rca.get('why_4', 'N/A'))
            st.write("**Why 5:**", existing_rca.get('why_5', 'N/A'))
            st.write("**Root Cause:**", existing_rca.get('root_cause', 'N/A'))
            
            st.subheader("Fishbone Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**People:**", existing_rca.get('people_factors', 'N/A'))
                st.write("**Process:**", existing_rca.get('process_factors', 'N/A'))
                st.write("**Equipment:**", existing_rca.get('equipment_factors', 'N/A'))
            with col2:
                st.write("**Materials:**", existing_rca.get('materials_factors', 'N/A'))
                st.write("**Environment:**", existing_rca.get('environment_factors', 'N/A'))
            
            st.subheader("Actions")
            st.write("**Corrective Actions:**", existing_rca.get('corrective_actions', 'N/A'))
            st.write("**Preventive Actions:**", existing_rca.get('preventive_actions', 'N/A'))
        else:
            st.info("No RCA found. Create one below.")
            
            with st.form("rca_form"):
                st.subheader("5-Why Analysis")
                why_1 = st.text_input("Why 1 - What happened?")
                why_2 = st.text_input("Why 2 - Why did it happen?")
                why_3 = st.text_input("Why 3 - Why did that happen?")
                why_4 = st.text_input("Why 4 - Why was that a factor?")
                why_5 = st.text_input("Why 5 - What is the root cause?")
                root_cause = st.text_area("Root Cause Summary")
                
                st.subheader("Fishbone Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    people = st.text_area("People Factors")
                    process = st.text_area("Process Factors")
                    equipment = st.text_area("Equipment Factors")
                with col2:
                    materials = st.text_area("Materials Factors")
                    environment = st.text_area("Environment Factors")
                
                st.subheader("Actions")
                corrective = st.text_area("Corrective Actions")
                preventive = st.text_area("Preventive Actions")
                completed_by = st.text_input("Completed By")
                
                submitted = st.form_submit_button("Submit RCA")
                
                if submitted:
                    data = {
                        "incident_id": incident_id,
                        "why_1": why_1,
                        "why_2": why_2,
                        "why_3": why_3,
                        "why_4": why_4,
                        "why_5": why_5,
                        "root_cause": root_cause,
                        "people_factors": people,
                        "process_factors": process,
                        "equipment_factors": equipment,
                        "materials_factors": materials,
                        "environment_factors": environment,
                        "corrective_actions": corrective,
                        "preventive_actions": preventive,
                        "completed_by": completed_by
                    }
                    result = call_api("POST", "/rca", data)
                    if result:
                        st.success("RCA created successfully!")
                        st.rerun()
    else:
        st.info("No downtime incidents available for RCA")

# ========== REPORTS PAGE ==========
elif page == "Reports":
    st.header("📊 Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Work Order Summary", "Downtime Summary", "Spare Parts Usage", "Asset History"]
    )
    
    if report_type == "Work Order Summary":
        work_orders = call_api("GET", "/work-orders")
        if work_orders:
            df = pd.DataFrame(work_orders)
            
            st.subheader("Work Order Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Work Orders", len(df))
            with col2:
                completed = len(df[df['status'] == 'COMPLETED'])
                st.metric("Completed", completed)
            with col3:
                open_wo = len(df[df['status'] == 'OPEN'])
                st.metric("Open", open_wo)
            
            # WO by type
            fig = px.pie(df, names='wo_type', title="Work Orders by Type")
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"work_orders_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    elif report_type == "Downtime Summary":
        incidents = call_api("GET", "/downtime")
        if incidents:
            df = pd.DataFrame(incidents)
            df['start_time'] = pd.to_datetime(df['start_time'])
            
            st.subheader("Downtime Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Incidents", len(df))
                st.metric("Total Downtime (hours)", f"{df['downtime_hours'].sum():.1f}")
            
            with col2:
                st.metric("Average Downtime (hours)", f"{df['downtime_hours'].mean():.1f}")
                critical = len(df[df['severity'] == 'CRITICAL'])
                st.metric("Critical Incidents", critical)
            
            # Downtime trend
            df_monthly = df.set_index('start_time').resample('M')['downtime_hours'].sum().reset_index()
            
            fig = px.line(
                df_monthly,
                x='start_time',
                y='downtime_hours',
                title="Monthly Downtime Trend",
                labels={'start_time': 'Month', 'downtime_hours': 'Downtime (hours)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"downtime_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

