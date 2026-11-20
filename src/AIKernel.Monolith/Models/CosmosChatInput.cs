using System.Text.Json;
using AIKernel.Kernel.Inference;
using AIKernel.Monolith.OpenAI;

namespace AIKernel.Monolith.Models;

internal sealed class CosmosChatInput : IDisposable
{
    internal string System { get; private set; } = "";
    internal List<CosmosChatTurn> History { get; } = [];
    internal IReadOnlyList<CosmosContent> User { get; private set; } = [];
    private readonly List<byte[]> pixels = [];
    private int patches;
    private long textLength;

    internal bool Read(IReadOnlyList<ChatMessage> messages,string architecture)
    {
        if (messages.Count is <1 or >32) return false;
        int i=0,temporal=architecture=="cosmos3-edge"?1:2;
        if (messages[0].Role is "system" or "developer") { System=messages[0].Content; textLength=System.Length; i++; }
        if ((messages.Count-i)%2!=1) return false;
        for (;i<messages.Count;i+=2)
        {
            var m=messages[i];
            if (m.Role!="user") return false;
            List<CosmosContent> content=[];
            if (m.Parts.ValueKind==JsonValueKind.Undefined)
            {
                if (!AddText(content,m.Content)) return false;
            }
            else
            {
                if (m.Parts.GetArrayLength() is <1 or >32) return false;
                foreach (var part in m.Parts.EnumerateArray())
                {
                    if (part.ValueKind!=JsonValueKind.Object || !part.TryGetProperty("type",out var type) || type.ValueKind!=JsonValueKind.String) return false;
                    switch(type.GetString())
                    {
                        case "text":
                            if (!part.TryGetProperty("text",out var text) || text.ValueKind!=JsonValueKind.String || !AddText(content,text.GetString()!)) return false;
                            break;
                        case "image_rgb24":
                            if (!TryFrame(part,0,true,out var image)) return false;
                            content.Add(new CosmosContent.Image(image!)); break;
                        case "video_rgb24":
                            if (!part.TryGetProperty("frames",out var frames) || frames.ValueKind!=JsonValueKind.Array || frames.GetArrayLength() is <1 or >64) return false;
                            List<CosmosRgbFrame> clip=[];
                            foreach (var frame in frames.EnumerateArray())
                            {
                                if (frame.ValueKind!=JsonValueKind.Object || !frame.TryGetProperty("seconds",out var time) || time.ValueKind!=JsonValueKind.Number ||
                                    !time.TryGetDouble(out double t) || !double.IsFinite(t) || t<0 || !TryFrame(frame,t,clip.Count%temporal==0,out var rgb)) return false;
                                if (clip.Count>0 && (t<clip[^1].TimestampSeconds || rgb!.Width!=clip[0].Width || rgb.Height!=clip[0].Height)) return false;
                                clip.Add(rgb!);
                            }
                            content.Add(new CosmosContent.Video(clip)); break;
                        default: return false;
                    }
                }
            }
            if (i+1==messages.Count) { User=content; break; }
            var answer=messages[i+1];
            if (answer.Role!="assistant") return false;
            textLength+=answer.Content.Length;
            History.Add(new(content,answer.Content));
        }
        return textLength<=32768;
    }

    private bool AddText(List<CosmosContent> content,string text)
    {
        textLength+=text.Length;
        if (textLength>32768 || string.IsNullOrWhiteSpace(text)) return false;
        content.Add(new CosmosContent.Text(text)); return true;
    }

    private bool TryFrame(JsonElement part,double seconds,bool countPatch,out CosmosRgbFrame? frame)
    {
        frame=null;
        static int Dimension(JsonElement part,string key) =>
            part.TryGetProperty(key,out var value) && value.ValueKind==JsonValueKind.Number && value.TryGetInt32(out int size) &&
            size is >=32 and <=16384 && size%32==0 ? size:0;
        int width=Dimension(part,"width"), height=Dimension(part,"height");
        int area=width/16*(height/16);
        if (area is <1 or >1024 || (countPatch && (patches+=area)>1024) ||
            !part.TryGetProperty("data",out var data) || data.ValueKind!=JsonValueKind.String) return false;
        string encoded=data.GetString()!;
        int length=width*height*3;
        if (encoded.Length!=4*((length+2)/3)) return false;
        var bytes=new byte[length]; pixels.Add(bytes);
        if (!Convert.TryFromBase64String(encoded,bytes,out int written) || written!=length) return false;
        frame=new(width,height,bytes,seconds);return true;
    }

    public void Dispose() { foreach (var bytes in pixels) Array.Clear(bytes); }
}
