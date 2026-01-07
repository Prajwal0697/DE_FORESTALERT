declare module 'mic-recorder-to-mp3' {
    interface MicRecorderOptions {
        bitRate?: number;
    }

    class MicRecorder {
        constructor(options?: MicRecorderOptions);
        
        start(): Promise<void>;
        
        stop(): any;
    }

    export = MicRecorder;
}
