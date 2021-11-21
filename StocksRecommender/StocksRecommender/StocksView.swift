//
//  StocksView.swift
//  StocksView
//

import SwiftUI

struct StocksView: View {
    var body: some View {
                VStack {
                    HStack(alignment: .firstTextBaseline) {
                        Image(systemName: "circle.fill")
                            .resizable()
                            .frame(width: 10, height:10)
                            .foregroundColor(Color.green)
                        VStack (alignment: .leading){
                            Text("Apple")
                                .font(.title)
                                .fontWeight(.bold)
                                .allowsTightening(true)
                        }
                        Spacer()
                        Text("+2.75")
                            .font(.footnote)
                            .fontWeight(.bold)
                            .foregroundColor(Color.green)
                            .allowsTightening(true)

                        Spacer()
                        Text("160.02")
                            .font(.title)
                            .fontWeight(.bold)
                    }

                    HStack(alignment: .firstTextBaseline) {
                        Image(systemName: "circle.fill")
                            .resizable()
                            .frame(width: 10, height:10)
                            .foregroundColor(Color.orange)
                        VStack (alignment: .leading){
                            Text("Microsoft")
                                .font(.title)
                                .fontWeight(.bold)
                                .allowsTightening(true)
                        }
                        Spacer()
                        Text("+1.84")
                            .font(.footnote)
                            .fontWeight(.bold)
                            .foregroundColor(Color.green)
                            .allowsTightening(true)
                        Spacer()
                        Text("343.02")
                            .font(.title)
                            .fontWeight(.bold)
                    }

                    HStack(alignment: .firstTextBaseline) {
                        Image(systemName: "circle.fill")
                            .resizable()
                            .frame(width: 10, height:10)
                            .foregroundColor(Color.orange)
                        VStack (alignment: .leading){
                            Text("Tesla")
                                .font(.title)
                                .fontWeight(.bold)
                                .allowsTightening(true)
                        }
                        Spacer()
                        Text("+40.68")
                            .font(.footnote)
                            .fontWeight(.bold)
                            .foregroundColor(Color.green)
                            .allowsTightening(true)
                        Spacer()
                        Text("1137.02")
                            .font(.title)
                            .fontWeight(.bold)
                    }

                    Spacer()

                }
                .padding(.all, 10)
            }

}

struct SecondView_Previews: PreviewProvider {
    static var previews: some View {
        StocksView()
    }
}
