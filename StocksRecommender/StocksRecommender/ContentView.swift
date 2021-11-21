//
//  ContentView.swift
//  StocksRecommender
//
//

import SwiftUI

struct ContentView: View {
    
    @State private var enableAirplaneMode = false

    var notificationMode = ["Low", "Medium", "High"]
    @State private var selectedMode = 0
    
    var body: some View {
        NavigationView {
            VStack {

            Form {

                Image("golden-coin")
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(height: 300)
                Section(header: Text("Risk Settings")){
                    
                    Picker(selection: $selectedMode, label: Text("Pick your risk level")) {
                        ForEach(0..<notificationMode.count) {
                            Text(self.notificationMode[$0])
                        }
                    }
                }
                
            }
            .padding(.top, 50)
            .listStyle(InsetGroupedListStyle())
            .navigationBarTitle("Welcome Sanchitha!")
            .overlay(
                ProfileView()
                    .padding(.trailing, 20)
                    .offset(x: 0, y: -50)
            , alignment: .topTrailing)

                NavigationLink(destination: StocksView()) {
                    Text("Today's recommended Stock picks")
                }

//                Button("Send Notification") {
//                          // 1.
//                          UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound])  {
//                                success, error in
//                                    if success {
//                                        print("authorization granted")
//                                    }
//
//                            }
                    let content = UNMutableNotificationContent()
                    content.title = "Good morning Sanchitha!"
                    content.subtitle = "Today's stock picks for you📈"
                    content.sound = UNNotificationSound.default

                    // show this notification five seconds from now
                    let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 5, repeats: false)

                    // choose a random identifier
                    let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: trigger)

                    // add our notification request
                    UNUserNotificationCenter.current().add(request)
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

struct ProfileView: View {
    var body: some View {
        Image("sanchu")
            .resizable()
            .clipped()
            .frame(width: 40, height: 40, alignment: .center)
            .clipShape(Circle())
            .overlay(Circle().stroke(Color.blue, lineWidth: 2.0))
    }
}
