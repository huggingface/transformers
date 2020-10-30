import speedtest
try:
    st = speedtest.Speedtest()
    server_names = []
    st.get_servers(server_names)
    downlink_bps = st.download()
    uplink_bps = st.upload()
    ping = st.results.ping
    up_mbps = round(uplink_bps/1000000,2)
    down_mbps = round(downlink_bps/1000000,2)
    print('speed test results:')
    print('ping: {} ms'.format(ping))
    print('upling: {} Mbps'.format(up_mbps))
    print('downling: {} Mbps'.format(down_mbps))
except:
    print("Speed test can't perform right now!")
