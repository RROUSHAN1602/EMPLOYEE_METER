import pandas as pd
import plotly.express as px
import re
import streamlit as st

# ---- STREAMLIT PAGE CONFIG ----
st.set_page_config(
        page_title="EMPLOYEE_METER",
        layout="wide",
        page_icon="📊"
    )

    # ---- STYLING FOR BETTER UI ----
st.markdown("""
        <style>
            .big-font { font-size:20px !important; }
            .st-emotion-cache-16txtl3 { padding: 1rem !important; }
            .st-emotion-cache-1outpf7 { border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
        </style>
    """, unsafe_allow_html=True)

    # ---- DASHBOARD TITLE ----
st.title("📊 EMPLOYEE_METER OMEGA FINANCIAL")


st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio("Go to", ["WANT TO COMBINE FILE", "EMPLOYEE_METER","About us"])

if page == "WANT TO COMBINE FILE":
    st.title("🏠COMBINE FILE")
    st.write("LET START COMBINING FILE!")


    st.title("📊 Sales Data Merger App")
    st.write("Upload six different CSV files, filter data, and generate a final merged file.")

    # Function to clean numeric values
    def clean_numeric(value):
        if isinstance(value, str):
            value = re.sub(r"\(.*?\)", "", value).strip()  # Remove percentages
            try:
                return float(value.replace(",", ""))  # Convert to float
            except ValueError:
                return None  # Return None if conversion fails
        return value

    # File uploaders for six datasets
    
    uploaded_files = {
        "NEGATIVE_SALES": st.sidebar.file_uploader("NEGATIVE Sales Report", type=["csv"]),
        "SOA_POSITIVE": st.sidebar.file_uploader("SOA Positive Sales Dataset", type=["csv"]),
        "POSITIVE_SALES": st.sidebar.file_uploader("POSITIVE Sales Dataset", type=["csv"]),
        "CLIENT_ADDITION": st.sidebar.file_uploader("CLIENT ADDITION Dataset", type=["csv"]),
        "DAILY_MEETINGS": st.sidebar.file_uploader("DAILY MEETINGS Dataset", type=["csv"]),
        "ACTIVATION": st.sidebar.file_uploader(" ACTIVATION Dataset", type=["csv","xlsx"]),
    }

    if all(uploaded_files.values()):  # Ensure all files are uploaded
        st.success("✅ All files uploaded successfully!")

        # Function to read CSV safely
        def read_csv_safely(file, skip_rows=8):
            try:
                df = pd.read_csv(file, skiprows=skip_rows, encoding='utf-8')
                return df if not df.empty else pd.DataFrame()
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                return pd.DataFrame()

        # Read datasets
        df_neg = read_csv_safely(uploaded_files["NEGATIVE_SALES"])
        df_pos1 = read_csv_safely(uploaded_files["SOA_POSITIVE"])
        df_pos2 = read_csv_safely(uploaded_files["POSITIVE_SALES"])
        df_client_add = read_csv_safely(uploaded_files["CLIENT_ADDITION"])
        df_meeting = read_csv_safely(uploaded_files["DAILY_MEETINGS"])
        df_ACTIVATION = read_csv_safely(uploaded_files["ACTIVATION"])

        # Process Negative Sales
        if not df_neg.empty:
            df_neg = df_neg[["Owner", "Product or Service", "Actual Closure Date", "Number of Deals", "Sum of Amount ( Actual Value )(In INR)"]]
            df_neg["Number of Deals"] = df_neg["Number of Deals"].apply(clean_numeric)
            df_neg["Sum of Amount ( Actual Value )(In INR)"] = df_neg["Sum of Amount ( Actual Value )(In INR)"].apply(clean_numeric)
            df_neg["Sum of Amount ( Actual Value )(In INR)"] *= -1  # Convert negative
            df_neg.rename(columns={"Actual Closure Date": "Date"}, inplace=True)

        # Process First Positive Sales
        if not df_pos1.empty:
            df_pos1 = df_pos1[["Owner", "Product or Service", "Created At", "Number of Deals", "Sum of Amount ( Actual Value )(In INR)"]]
            df_pos1["Number of Deals"] = df_pos1["Number of Deals"].apply(clean_numeric)
            df_pos1["Sum of Amount ( Actual Value )(In INR)"] = df_pos1["Sum of Amount ( Actual Value )(In INR)"].apply(clean_numeric)
            df_pos1.rename(columns={"Created At": "Date"}, inplace=True)

        # Process Second Positive Sales
        if not df_pos2.empty:
            df_pos2 = df_pos2[["Owner", "Product or Service", "Pipeline", "Number of Deals", "Sum of Amount ( Actual Value )(In INR)"]]
            df_pos2["Number of Deals"] = df_pos2["Number of Deals"].apply(clean_numeric)
            df_pos2["Sum of Amount ( Actual Value )(In INR)"] = df_pos2["Sum of Amount ( Actual Value )(In INR)"].apply(clean_numeric)
            df_pos2.rename(columns={"Pipeline": "Date"}, inplace=True)

        # Merge Sales Data
        final_df = pd.concat([df_neg, df_pos1, df_pos2], ignore_index=True)

        # Add Extra Columns
        extra_columns = ["Number of CLIENT TYPE", "Number of meetings", "Status", "Activation", "Specific Task"]
        for col in extra_columns:
            final_df[col] = None

        # Process Client Addition
        if not df_client_add.empty:
            df_client_add = df_client_add[["Owner", "Number of Deals", "Number of CLIENT TYPE"]]
            df_client_add["Number of Deals"] = df_client_add["Number of Deals"].apply(clean_numeric)
            df_client_add["Number of CLIENT TYPE"] = df_client_add["Number of CLIENT TYPE"].apply(clean_numeric)
            df_client_add = df_client_add.reindex(columns=final_df.columns, fill_value=None)
            final_df = pd.concat([final_df, df_client_add], ignore_index=True)

        # Process Daily Meeting
        if not df_meeting.empty:
            df_meeting = df_meeting[["Created By", "Number of meetings", "Status"]]
            df_meeting.rename(columns={"Created By": "Owner"}, inplace=True)
            df_meeting["Number of meetings"] = df_meeting["Number of meetings"].apply(clean_numeric)
            df_meeting = df_meeting.reindex(columns=final_df.columns, fill_value=None)
            final_df = pd.concat([final_df, df_meeting], ignore_index=True)

        # Process Activation & Specific Task
        if not df_ACTIVATION.empty:
            df_ACTIVATION.columns = df_ACTIVATION.columns.str.strip().str.lower()  # Clean column names
            rename_map = {"owner": "Owner", "activation": "Activation", "Specific Task": "Specific Task"}
            df_ACTIVATION.rename(columns=rename_map, inplace=True)

            # Ensure required columns exist
            for col in ["Owner", "Activation", "Specific Task"]:
                if col not in df_ACTIVATION.columns:
                    df_ACTIVATION[col] = None

            df_ACTIVATION = df_ACTIVATION[["Owner", "Activation", "Specific Task"]]
            df_ACTIVATION = df_ACTIVATION.reindex(columns=final_df.columns, fill_value=None)
            final_df = pd.concat([final_df, df_ACTIVATION], ignore_index=True)

        # Sidebar Filter by Owner
        st.sidebar.header("🔍 Filter Data")
        if "Owner" in final_df.columns:
            owners = final_df["Owner"].dropna().unique().tolist()
            selected_owners = st.sidebar.multiselect("Select Owners", options=owners, default=owners)

            if selected_owners:
                final_df = final_df[final_df["Owner"].isin(selected_owners)]

        # Display Processed Data
        st.write("### Final Merged Data")
        st.dataframe(final_df)

        # Download Button
        st.sidebar.header("📥 Download Merged File")
        if not final_df.empty:
            final_csv = final_df.to_csv(index=False, encoding="utf-8").encode("utf-8")
            st.sidebar.download_button(label="⬇️ Download CSV", data=final_csv, file_name="merged_sales_data.csv", mime="text/csv")

    else:
        st.warning("⚠️ Please upload all six required CSV files.")


elif page == "EMPLOYEE_METER":
    st.title("📄EMPLOYEE_METER")
    st.write("wElCoMe TO OMEGA empLOYEE meTER ")
    
    # ---- FILE UPLOAD FEATURE ----
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
    # hello everyone my name is himshikha mahant 
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df["Sum of Amount ( Actual Value )(In INR)"] = pd.to_numeric(df["Sum of Amount ( Actual Value )(In INR)"], errors="coerce")

        # ritik roushan is my best friend 
        st.success("✅ File uploaded successfully!")

        # ---- SIDEBAR FILTER ----
        st.sidebar.header("🔍 Filter Data")
        owners = df["Owner"].dropna().unique()
        selected_owners = st.sidebar.multiselect("Select Owners", options=owners, default=owners)

        filtered_df = df[df["Owner"].isin(selected_owners)] if selected_owners else df

        # ---- BUSINESS POINTS CALCULATION ----
        points_rules = {
            "MUTUAL FUND ( DEBT)": 6 / 100000,
            "MUTUAL FUND ( EQUITY )": 30 / 100000,
            "MUTUAL FUND SIP ( DEBT )": 120 / 100000,
            "MUTUAL FUND SIP (EQUITY)": 800 / 100000,
            "PORTFOLIO MANAGEMENT SERVICES": 50 / 100000,
            "LIFE INSURANCE": 400 / 100000,
            "GENERAL INSURANCE": 0 / 100000,
            "HEALTH INSURANCE": 350 / 100000,
            "ARN TRANSFER-EQUITY": 0 / 100000,
            "ARN TRANSFER-DEBT": 0 / 100000
        }

        filtered_df["Business Points"] = filtered_df["Product or Service"].map(points_rules) * filtered_df["Sum of Amount ( Actual Value )(In INR)"]
        business_points_data = filtered_df.groupby("Owner")["Business Points"].sum().reset_index().fillna(0)
        business_points_data["Rank"] = business_points_data["Business Points"].rank(method='dense', ascending=False).astype(int)
        business_points_data = business_points_data.sort_values(by="Rank")

        # ---- TOTAL BUSINESS POINTS & NET AMOUNT ----
        total_business_points = business_points_data["Business Points"].sum()
        total_net_amount = filtered_df["Sum of Amount ( Actual Value )(In INR)"].sum()

        # ---- DISPLAY SELECTED OWNER(S) RANK POSITION ----
        rank_info = []
        # god bless you my dear
        for owner in selected_owners:
            owner_rank_row = business_points_data[business_points_data["Owner"] == owner]
            if not owner_rank_row.empty:
                rank_val = int(owner_rank_row["Rank"].values[0])
                rank_info.append(f"{owner}: Rank {rank_val}")
        rank_display = " | ".join(rank_info) if rank_info else "No rank available"

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; 
                        align-items: center; padding: 20px; margin-bottom: 20px; 
                        background-color: #f1f3f6; border-radius: 15px; 
                        border: 2px solid #ccc; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
                        font-size: 24px; font-weight: bold; color: #333;">
                <div>🏆 <span style="color:#ff5733;">Rank Position:</span> {rank_display}</div>
                <div>💼 Total Business Points: <span style="color:#007bff;">{total_business_points:,.2f}</span></div>
                <div>💰 Net Amount: <span style="color:#28a745;">₹{total_net_amount:,.2f}</span></div>
            </div>
        """, unsafe_allow_html=True)

        # ---- AMOUNT CATEGORIZATION ----
        amount_categories = filtered_df.groupby("Product or Service")["Sum of Amount ( Actual Value )(In INR)"].sum().reset_index()
        st.write(f"### 💰 Amount Categorization")
        st.dataframe(amount_categories)

        # ---- CHARTS WITHOUT X-AXIS LABELS ----
        fig1 = px.bar(amount_categories, x="Product or Service", y="Sum of Amount ( Actual Value )(In INR)",
                    title="💰 Amount Categorization", text="Sum of Amount ( Actual Value )(In INR)")
        fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig1.update_layout(xaxis_title="", xaxis_showticklabels=False)

        # add some graphics  
        fig2 = px.bar(business_points_data, x="Owner", y="Business Points",
                    title="🏆 Business Points Ranking", text="Business Points", color="Owner")
        fig2.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig2.update_layout(xaxis_title="", xaxis_showticklabels=False)

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1)
        col2.plotly_chart(fig2)

        # ---- AUM & SIP CATEGORIZATION ----
        def categorize_aum_sip(data):
            aum_df = data[data["Product or Service"].str.contains("EQUITY|DEBT|PORTFOLIO MANAGEMENT SERVICES|PMS", case=False, na=False) &
                        ~data["Product or Service"].str.contains("SIP", case=False, na=False)]
            sip_df = data[data["Product or Service"].str.contains("SIP", case=False, na=False)]
            aum = aum_df.groupby("Owner")["Sum of Amount ( Actual Value )(In INR)"].sum().reset_index().rename(
                columns={"Sum of Amount ( Actual Value )(In INR)": "AUM Amount"})
            sip = sip_df.groupby("Owner")["Sum of Amount ( Actual Value )(In INR)"].sum().reset_index().rename(
                columns={"Sum of Amount ( Actual Value )(In INR)": "SIP Amount"})
            return pd.merge(aum, sip, on="Owner", how="outer").fillna(0)

        aum_sip_data = categorize_aum_sip(filtered_df)
        st.write("### 💰 AUM & SIP Distribution")
        st.dataframe(aum_sip_data)

        fig3 = px.bar(aum_sip_data, x="Owner", y=["AUM Amount", "SIP Amount"],
                    title="💰 AUM & SIP Distribution", barmode="group", text_auto=True)
        fig3.update_layout(xaxis_title="", xaxis_showticklabels=False)
        st.plotly_chart(fig3)

        # ---- PERFORMANCE METRICS ----
        def calculate_performance_metrics(data):
            perf = data.groupby("Owner").agg({
                "Sum of Amount ( Actual Value )(In INR)": "sum",
                "Number of CLIENT TYPE": "sum",
                "Number of meetings": "sum",
                "Activation": "sum",
                "Specific Task": "sum"
            }).reset_index()
            deals = data.groupby("Owner").size().reset_index(name="Number of Deals")
            return pd.merge(perf, deals, on="Owner", how="left")

        performance_metrics_data = calculate_performance_metrics(filtered_df)
        st.write("### 📈 Performance Metrics")
        st.dataframe(performance_metrics_data)

        # ---- FINAL STRUCTURED MATRIX ----
        def create_final_matrix(aum_sip_data, perf_data, bp_data):
            final = pd.merge(aum_sip_data, perf_data, on="Owner", how="left")
            final = pd.merge(final, bp_data, on="Owner", how="left").fillna(0)
            final = final.rename(columns={
                "Owner": "CANDIDATE",
                "Number of meetings": "Client Meeting",
                "Number of CLIENT TYPE": "New Client Addition",
                "AUM Amount": "AUM",
                "SIP Amount": "SIP"
            })
            final.insert(0, "SL NO", range(1, len(final) + 1))
            return final

        final_matrix = create_final_matrix(aum_sip_data, performance_metrics_data, business_points_data)
        st.write("### 📊 Final Structured Matrix")
        st.dataframe(final_matrix)

    else:
        st.warning("⚠ Please upload a CSV file to proceed.")

elif page == "About us":
        
        st.title("📄NICE TO SEE YOU HERE")
        st.write("THANK YOU FOR VISITS.")
  
        image_url = "https://media.licdn.com/dms/image/v2/C510BAQGi9wEg7Gspdw/company-logo_200_200/company-logo_200_200/0/1630601866330?e=2147483647&v=beta&t=KJUNX1I5DYg6eS5MluO2DrzWyC42-elqkeuNByTE-Uw"
        
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="{image_url}" style="width: 220px; height: 220px;">
            </div>
            """,
            unsafe_allow_html=True
        )

    
    
