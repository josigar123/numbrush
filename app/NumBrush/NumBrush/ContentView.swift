import SwiftUI

struct ContentView: View {
    var body: some View {
        CanvasView { image in
            print("Got image: \(image.size)")
            // PredictionService call goes here later
        }
    }
}

#Preview {
    ContentView()
}
