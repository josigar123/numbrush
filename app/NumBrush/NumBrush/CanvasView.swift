import SwiftUI

struct CanvasView: View {
    @State private var lines: [Line] = []
    @State private var currentLine: Line = Line()
    var onSubmit: (UIImage) -> Void

    var body: some View {
        VStack {
            Canvas { context, size in
                for line in lines + [currentLine] {
                    var path = Path()
                    path.addLines(line.points)
                    context.stroke(
                        path,
                        with: .color(.white),
                        style: StrokeStyle(lineWidth: 20, lineCap: .round, lineJoin: .round)
                    )
                }
            }
            .background(Color.black)
            .frame(width: 280, height: 280)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        currentLine.points.append(value.location)
                    }
                    .onEnded { _ in
                        lines.append(currentLine)
                        currentLine = Line()
                    }
            )

            HStack {
                Button("Clear") {
                    lines = []
                }
                .padding()

                Button("Predict") {
                    onSubmit(snapshot())
                }
                .padding()
            }
        }
    }

    private func snapshot() -> UIImage {
        let renderer = ImageRenderer(
            content: Canvas { context, size in
                context.fill(
                    Path(CGRect(origin: .zero, size: size)),
                    with: .color(.black)
                )
                for line in lines {
                    var path = Path()
                    path.addLines(line.points)
                    context.stroke(
                        path,
                        with: .color(.white),
                        style: StrokeStyle(lineWidth: 20, lineCap: .round, lineJoin: .round)
                    )
                }
            }
            .frame(width: 280, height: 280)
        )
        return renderer.uiImage ?? UIImage()
    }
}

struct Line {
    var points: [CGPoint] = []
}
